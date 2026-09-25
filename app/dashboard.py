"""The analyst's view of an investigation.

Everything on the screen is read back from TigerGraph, not from the answer files: the case
vertex, its event trail, the transactions and cards it touched, and the closed cases it
drew on. That is the point of writing cases to the graph — an analyst opens the case the
same way the next investigation retrieves it.

Four things the round asks to be visible are each given their own place: how the case
progressed, what evidence it rests on, how certain the agent is, and what it recommends
with the approval each action needs. The first tab draws the case where it sits in the
graph: the card, the transactions, the devices and the linked cards around it.

Dressed in the Hacker House Goa 2026 theme; the styles are in app/theme.css.

Run with:  uv run streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import Any, TypeVar

import streamlit as st

from fraudtrail.answer.schema import ActionItem, Answer
from fraudtrail.config import ConfigError, Settings, load_settings
from fraudtrail.graph.case_graph import CaseGraph, Kind, draw
from fraudtrail.graph.client import GraphClient, GraphError
from fraudtrail.graph.memory import GraphMemory
from fraudtrail.investigate import constants as ic
from fraudtrail.policy.actions import Route
from fraudtrail.wording import counted

REPO = Path(__file__).resolve().parent.parent
CASES_DIR = REPO / "cases"
THEME_CSS = Path(__file__).resolve().parent / "theme.css"
GRAPH_PAGE = Path(__file__).resolve().parent / "case_graph.html"

T = TypeVar("T")

# Savanna's auto-stop pauses an idle workspace. Its first request back is answered with a
# "Starting workspace" page, or a 5xx, while it resumes, which takes a minute or two. The
# dashboard waits that long rather than showing a visitor a traceback.
WAKING_MARKERS = ("Starting workspace", "500 Server Error", "502", "503", "504")
GRAPH_WAKE_TIMEOUT_S = 180
GRAPH_WAKE_POLL_S = 8

# How much of an unexpected graph error to show, once markup is stripped from it.
MAX_ERROR_CHARS = 200

# The exam window, for the ring search.
RING_FROM = "2016-11-01 00:00:00"
RING_TO = "2016-12-31 23:59:59"

# Label propagation rounds. A ring is a star of cards around one or two devices, so labels
# settle within a few hops; six is comfortably past that and still bounded.
LABEL_ROUNDS = 6

# Share of a device's transactions behind an anonymous proxy. In the exam window 71 devices
# pass the investigation's three ring tests; exactly one of them, the device behind HHG-014,
# is behind an anonymous proxy at all (on every transaction). The other 70 are ordinary
# Apple and Windows configurations at zero.
RING_MIN_ANONYMOUS_SHARE = 0.5

# How many rings the view lists, and how many cards each one shows before trimming.
MAX_RINGS_SHOWN = 10
MAX_RING_CARDS_SHOWN = 40

# Height of the investigation graph. Tall enough for a ring of twenty cards around a device
# to spread out, short enough to leave the tabs in view on a laptop screen.
GRAPH_HEIGHT_PX = 600

# How many connected cards and device profiles the evidence tab lists before trimming.
MAX_CONNECTED_SHOWN = 12
MAX_PROFILES_SHOWN = 3

ROUTE_HELP = {
    Route.AUTO: "the agent may do this without asking",
    Route.L1: "a fraud analyst must approve",
    Route.L2: "a fraud manager must approve",
}

# ?view= values, and the tab each opens.
VIEWS = ["Case", "Queue", "Rings"]
VIEW_LABELS = {"Case": "Case Investigation", "Queue": "Approval Queue", "Rings": "Ring Finder"}

# Trusted markup, written here rather than escaped: the entity is the multiplication sign.
EVENT_HTML = "TigerGraph &times; HH Goa"
TASK = "Agentic fraud investigation"

# A magnifier over a card: the mark beside the name, drawn inline so nothing is fetched.
MARK = (
    "<svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
    "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
    "<rect x='2' y='5' width='15' height='11' rx='2'/><path d='M2 9h15'/>"
    "<circle cx='17.5' cy='16.5' r='3.5'/><path d='m20 19 2 2'/></svg>"
)


@st.cache_resource
def graph_memory() -> GraphMemory:
    return GraphMemory(GraphClient.from_env())


@st.cache_resource
def graph_client() -> GraphClient:
    return GraphClient.from_env()


def cases_stamp() -> float:
    """When the answer files last changed.

    The caches below take it as an argument, so a new run of the agent shows up on the next
    page load instead of being served from a cache filled before it.
    """
    return max((p.stat().st_mtime for p in CASES_DIR.glob("HHG-*.json")), default=0.0)


@st.cache_data
def load_answers(stamp: float) -> dict[str, Answer]:
    answers: dict[str, Answer] = {}
    for path in sorted(CASES_DIR.glob("HHG-*.json")):
        answers[path.stem] = Answer.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return answers


@st.cache_data(show_spinner="Reading the case from the graph…")
def read_case(case_id: str, stamp: float) -> dict[str, Any]:
    return dict(graph_memory().read(case_id))


@st.cache_data(show_spinner="Reading the case's neighbourhood from the graph…")
def read_neighbourhood(case_id: str, stamp: float) -> dict[str, Any]:
    return dict(graph_memory().neighbourhood(case_id))


@st.cache_data(show_spinner="Running label propagation over shared devices…")
def find_rings(min_cards: int) -> list[tuple[list[str], list[str]]]:
    """Communities of cards that share a device, found by the graph algorithm.

    Each ring comes back as its cards and the device profiles that link them.
    """
    blocks = graph_client().run_query(
        "ring_components",
        {
            "from_ts": RING_FROM,
            "to_ts": RING_TO,
            "rounds": LABEL_ROUNDS,
            "min_ring_cards": min_cards,
            # The tests the investigation applies to a shared device, and the proxy test.
            "min_device_cards": ic.SHARED_ORIGIN_MIN_CARDS,
            "max_device_cards": ic.SHARED_ORIGIN_MAX_CARDS,
            "min_new_share": ic.SHARED_ORIGIN_MIN_NEW_DEVICE_SHARE,
            "min_anonymous_share": RING_MIN_ANONYMOUS_SHARE,
        },
    )
    merged: dict[str, Any] = {}
    for block in blocks:
        if isinstance(block, dict):
            merged.update(block)
    components = merged.get("components")
    on_device = merged.get("cards_on_device")
    if not isinstance(components, dict):
        return []
    devices = on_device if isinstance(on_device, dict) else {}
    rings: list[tuple[list[str], list[str]]] = []
    for cards in components.values():
        if not isinstance(cards, list) or len(cards) < min_cards:
            continue
        members = {str(c) for c in cards}
        linking = sorted(
            str(profile)
            for profile, on in devices.items()
            if isinstance(on, list) and members & {str(c) for c in on}
        )
        rings.append((sorted(members), linking))
    rings.sort(key=lambda ring: len(ring[0]), reverse=True)
    return rings


def from_graph(read: Callable[[], T]) -> T | None:
    """A graph read that waits out a workspace auto-stop has paused, instead of failing."""
    notice = st.empty()
    deadline = time.monotonic() + GRAPH_WAKE_TIMEOUT_S
    while True:
        try:
            result = read()
        except ConfigError as exc:
            notice.warning(str(exc))
            return None
        except GraphError as exc:
            reason = str(exc)
            waking = any(marker in reason for marker in WAKING_MARKERS)
            if waking and time.monotonic() < deadline:
                notice.info(
                    "The TigerGraph workspace is waking up after auto-stop. This takes a "
                    "minute or two, and the case will appear here on its own."
                )
                time.sleep(GRAPH_WAKE_POLL_S)
                continue
            if waking:
                notice.warning("The graph is still starting up. Refresh the page in a minute.")
            else:
                plain = re.sub(r"<[^>]+>", " ", reason)
                notice.warning(f"The graph did not answer: {plain[:MAX_ERROR_CHARS]}")
            return None
        notice.empty()
        return result


def money(value: float) -> str:
    return f"${value:,.2f}"


def md(text: str) -> str:
    """Markdown reads a pair of dollar signs as mathematics, and every amount here has one."""
    return text.replace("$", r"\$")


def label(text: str) -> None:
    """A section label in the event's wide-tracked mono, like the playground's TRY ASKING."""
    st.html(f"<div class='ft-label'>{escape(text)}</div>")


def apply_theme() -> None:
    st.html(f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>")


def show_masthead(settings: Settings, answers: dict[str, Answer]) -> None:
    in_graph = sum(1 for a in answers.values() if a.case.written_to_graph)
    st.html(
        "<header class='ft-header'>"
        f"<div class='ft-brand'>{MARK}<h1 class='ft-title'>FRAUDTRAIL</h1>"
        "<span class='ft-pill'>HH Goa 2026</span></div>"
        "<div class='goa-wordmark' aria-hidden='true'>"
        "<span>HACKER</span><span class='goa-chip'>गोवा</span><span>HOUSE</span></div>"
        "<div class='ft-status'>"
        f"<span class='ft-badge'>graph {escape(settings.tigergraph.graph)}</span>"
        f"<span class='ft-badge'>{counted(len(answers), 'case')}</span>"
        f"<span class='ft-badge'>{in_graph} of {len(answers)} in the graph</span>"
        "</div></header>"
    )


def show_navigation() -> str:
    # ?view=Queue or ?case=HHG-014 opens that view directly, so a case can be shared as a link.
    requested = st.query_params.get("view", VIEWS[0])
    nav, event = st.columns([5, 3])
    with nav, st.container(key="nav"):
        view = st.radio(
            "View",
            VIEWS,
            index=VIEWS.index(requested) if requested in VIEWS else 0,
            horizontal=True,
            format_func=lambda v: VIEW_LABELS[v],
            label_visibility="collapsed",
        )
    with event:
        st.html(
            f"<div class='ft-event'><span class='ft-team'>{EVENT_HTML}</span> | "
            f"<span class='ft-dates'>{escape(TASK)}</span></div>"
        )
    return str(view)


def stat(name: str, value: str, tone: str = "") -> str:
    return f"<div class='ft-stat {tone}'><span>{escape(name)}</span><b>{escape(value)}</b></div>"


def show_case_header(answer: Answer) -> None:
    verdict = answer.case.verdict.value
    pattern = answer.case.pattern.value.replace("_", " ")
    text, figures = st.columns([1.3, 1], gap="large")
    with text:
        st.html(
            f"<div class='ft-case-head'><span class='ft-case-id'>{escape(answer.case_id)}</span>"
            f"<span class='ft-verdict {escape(verdict)}'>{escape(verdict)}</span>"
            f"<span class='ft-pattern'>{escape(pattern)}</span></div>"
        )
        st.write(md(answer.case.summary))
        if answer.case.pattern_description:
            st.info(md(f"**Undocumented pattern.** {answer.case.pattern_description}"))
    with figures:
        tone = "hot" if verdict == "fraud" else "cool"
        written = "yes" if answer.case.written_to_graph else "no"
        st.html(
            "<div class='ft-stats'>"
            + stat("Fraud probability", f"{answer.case.fraud_probability:.2f}", tone)
            + stat("Exposure", money(answer.case.exposure_usd))
            + stat("Connected cards", str(len(answer.case.connected_card_ids)))
            + stat("Evidence queries", str(answer.tool_calls))
            + "</div><div class='ft-strip'>"
            f"<div><span>In graph</span><b>{written}</b></div>"
            f"<div><span>Latency</span><b>{answer.latency_s:.2f}s</b></div>"
            f"<div><span>Tokens</span><b>{answer.tokens:,}</b></div>"
            f"<div><span>Asked</span><b>{len(answer.evidence_requests)}</b></div>"
            f"<div><span>Report</span><b>{'filed' if answer.sar.file else 'none'}</b></div>"
            "</div>"
        )


def show_case_pack(answers: dict[str, Answer], current: str) -> None:
    """Every exam case as a link, like the playground's suggested questions."""
    label("Case pack")
    rows = []
    for case_id, answer in answers.items():
        verdict = answer.case.verdict.value
        active = " active" if case_id == current else ""
        rows.append(
            f"<a class='ft-chip {escape(verdict)}{active}' href='?case={escape(case_id)}'>"
            f"<span>{escape(case_id)}</span><i>{answer.case.fraud_probability:.2f}</i></a>"
        )
    fraud = sum(1 for a in answers.values() if a.case.verdict.value == "fraud")
    reports = sum(1 for a in answers.values() if a.sar.file)
    st.html(
        f"<div class='ft-chips'>{''.join(rows)}</div>"
        f"<div class='ft-legend'>{fraud} fraud · {len(answers) - fraud} legitimate · "
        f"{counted(reports, 'report')}</div>"
    )


def action_card(item: ActionItem) -> str:
    route = item.route.value
    return (
        "<div class='ft-action'>"
        f"<span class='a'>{escape(item.action.value)}</span>"
        f"<span class='r {escape(route)}'>{escape(route)}</span>"
        f"<span class='h'>{escape(ROUTE_HELP.get(item.route, ''))}</span>"
        f"<div class='w'>{escape(item.reason)}</div></div>"
    )


def show_actions(answer: Answer) -> None:
    st.subheader("Recommended actions")
    st.caption(
        "The agent recommends; it executes nothing that needs a human. "
        "Each action carries the approval its route requires."
    )
    before, after = st.columns(2)
    with before:
        label("Before more evidence was requested")
        st.html("".join(action_card(i) for i in answer.next_best_actions.initial))
    with after:
        label("After the response came back")
        st.html("".join(action_card(i) for i in answer.next_best_actions.final))
    st.markdown(md(f"**What changed:** {answer.next_best_actions.what_changed}"))


def show_uncertainty(answer: Answer) -> None:
    st.subheader("Uncertainty")
    st.progress(
        min(max(answer.case.fraud_probability, 0.0), 1.0),
        text=f"fraud probability {answer.case.fraud_probability:.2f}",
    )
    st.markdown(md(f"**Why the investigation stopped:** {answer.stop_reason}"))
    for request in answer.evidence_requests:
        st.warning(
            f"**Asked for more evidence** after step {request.asked_after_step}: "
            f"`{request.type.value}`\n\n"
            f"Assumed response: _{md(request.assumed_response)}_"
        )
    if not answer.evidence_requests:
        st.caption("The evidence settled the question without asking anyone.")


def show_evidence(answer: Answer) -> None:
    st.subheader("Evidence")
    rows = [
        {
            "claim": item.claim,
            "source": item.source.value,
            "ref": item.ref,
            "entities": ", ".join(item.entity_ids[:4]),
        }
        for item in answer.case.evidence
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    connected = answer.case.connected_card_ids
    if connected:
        st.markdown(
            f"**Connected cards ({len(connected)}):** "
            + ", ".join(f"`{c}`" for c in connected[:MAX_CONNECTED_SHOWN])
            + ("…" if len(connected) > MAX_CONNECTED_SHOWN else "")
        )
    for profile in answer.case.connected_device_profiles[:MAX_PROFILES_SHOWN]:
        st.code(profile, language=None)
    if answer.case.similar_prior_cases:
        st.markdown(
            "**Prior cases retrieved from memory:** "
            + ", ".join(f"`{c}`" for c in answer.case.similar_prior_cases)
        )


def graph_page(graph: CaseGraph) -> str:
    # The data goes in as JSON inside a script; escaping "<" keeps any value from closing it.
    data = json.dumps(graph.for_drawing()).replace("<", r"\u003c")
    return GRAPH_PAGE.read_text(encoding="utf-8").replace("__GRAPH__", data)


def show_graph(case_id: str) -> None:
    st.subheader("Investigation graph")
    st.caption(
        "Read from TigerGraph: the case, the card it was opened on and its owner, the "
        "transactions it flagged with their device and billing region, the linked cards, and "
        "the closed cases it drew on. Solid lines are evidence in the graph; dashed yellow "
        "lines are the links this investigation wrote."
    )
    stored = from_graph(lambda: read_neighbourhood(case_id, cases_stamp()))
    if stored is None:
        return
    graph = draw(stored)
    if not graph.nodes:
        st.warning("This case is not in the graph yet. Run scripts/run_agent.py.")
        return
    txns = graph.count(Kind.FLAGGED, Kind.TRANSACTION)
    cards = graph.count(Kind.CARD, Kind.OWNER_CARD, Kind.LINKED_CARD)
    counts = [
        counted(txns, "transaction"),
        counted(cards, "card"),
        counted(graph.count(Kind.DEVICE), "device profile"),
        counted(graph.count(Kind.CLOSED_CASE), "closed case"),
    ]
    st.html(
        "<div class='ft-counts'>"
        + "".join(f"<span class='ft-badge'>{escape(c)}</span>" for c in counts)
        + "</div>"
    )
    st.iframe(graph_page(graph), height=GRAPH_HEIGHT_PX)


def show_progression(case_id: str) -> None:
    st.subheader("Case progression")
    st.caption("Read back from the graph: every step this investigation took, in order.")
    stored = from_graph(lambda: read_case(case_id, cases_stamp()))
    if stored is None:
        return
    events = stored.get("events")
    if not isinstance(events, list) or not events:
        st.warning("This case is not in the graph yet. Run scripts/run_agent.py.")
        return
    ordered = sorted(events, key=lambda e: int(e.get("attributes", {}).get("step", 0)))
    steps = []
    for event in ordered:
        a = event.get("attributes", {})
        steps.append(
            "<div class='ft-step'>"
            f"<span class='n'>{escape(str(a.get('step', '')))}</span>"
            f"<span class='k'>{escape(str(a.get('kind', '')).replace('_', ' '))}</span>"
            f"<span class='d'>{escape(str(a.get('detail', '')))}</span></div>"
        )
    st.html("".join(steps))


def show_report(answer: Answer) -> None:
    st.subheader("Suspicious activity report")
    if not answer.sar.file:
        st.success(md(f"No report required. {answer.sar.reason}"))
        return
    st.error(md(f"**Report required.** {answer.sar.reason}"))
    st.html(
        "<div class='ft-stats'>"
        + stat("Total amount", money(answer.sar.total_amount_usd), "hot")
        + stat("Subjects", str(len(answer.sar.subjects)))
        + "</div>"
    )
    st.markdown(f"**Activity dates:** {' to '.join(answer.sar.activity_dates)}")
    st.markdown(f"**Subjects:** {', '.join(answer.sar.subjects)}")
    st.markdown("**Narrative**")
    st.write(md(answer.sar.narrative))


def show_case_view(answers: dict[str, Answer]) -> None:
    case_ids = list(answers)
    requested = st.query_params.get("case", "")
    main, side = st.container(key="caseview").columns([3.3, 1], gap="medium")
    with side, st.container(key="panel_pack"):
        label("Case")
        case_id = str(
            st.selectbox(
                "Case",
                case_ids,
                index=case_ids.index(requested) if requested in case_ids else 0,
                format_func=lambda c: f"{c} — {answers[c].case.verdict.value}",
                label_visibility="collapsed",
            )
        )
        st.query_params["case"] = case_id
        show_case_pack(answers, case_id)
    answer = answers[case_id]
    with main:
        with st.container(key="panel_case"):
            show_case_header(answer)
        with st.container(key="panel_tabs"):
            graph_case_id = answer.case.graph_case_id or f"CASE-{case_id}"
            tabs = st.tabs(["Graph", "Progression", "Evidence", "Uncertainty", "Actions", "Report"])
            with tabs[0]:
                show_graph(graph_case_id)
            with tabs[1]:
                show_progression(graph_case_id)
            with tabs[2]:
                show_evidence(answer)
            with tabs[3]:
                show_uncertainty(answer)
            with tabs[4]:
                show_actions(answer)
            with tabs[5]:
                show_report(answer)


def show_queue(answers: dict[str, Answer]) -> None:
    with st.container(key="panel_queue"):
        st.subheader("Approval queue")
        st.caption("Everything the agent recommends that a human must approve before it happens.")
        rows = [
            {
                "case": case_id,
                "verdict": answer.case.verdict.value,
                "action": item.action.value,
                "route": item.route.value,
                "exposure": money(answer.case.exposure_usd),
                "reason": item.reason,
            }
            for case_id, answer in answers.items()
            for item in answer.next_best_actions.final
            if item.route is not Route.AUTO
        ]
        rows.sort(key=lambda r: (r["route"], r["case"]), reverse=True)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(f"{len(rows)} actions awaiting approval across {len(answers)} cases.")


def show_rings() -> None:
    with st.container(key="panel_rings"):
        st.subheader("Rings in the exam window")
        st.caption(
            "Label propagation over cards that share a device profile, run in the graph. "
            "Nothing here is told what to look for. A device takes part only if it could be a "
            "ring: a fully specified profile, on several cards but not hundreds, new on nearly "
            "every account it touches, and behind an anonymous proxy."
        )
        min_cards = st.slider("Smallest ring to show", 3, 25, 8)
        if not st.button("Run the graph algorithm"):
            return
        rings = from_graph(lambda: find_rings(min_cards))
        if rings is None:
            return
        st.write(f"{counted(len(rings), 'ring')} of {min_cards} or more cards")
        for index, (cards, devices) in enumerate(rings[:MAX_RINGS_SHOWN]):
            linked_by = devices[0] if len(devices) == 1 else counted(len(devices), "device profile")
            with st.expander(
                f"{counted(len(cards), 'card')} linked by {linked_by}", expanded=not index
            ):
                st.write(", ".join(f"`{c}`" for c in cards[:MAX_RING_CARDS_SHOWN]))
                for device in devices:
                    st.code(device, language=None)


def main() -> None:
    st.set_page_config(
        page_title="FraudTrail · HH Goa 2026",
        page_icon=":material/policy:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_theme()
    settings = load_settings()
    answers = load_answers(cases_stamp())
    show_masthead(settings, answers)
    view = show_navigation()
    if view == "Queue":
        show_queue(answers)
    elif view == "Rings":
        show_rings()
    else:
        show_case_view(answers)


main()
