"""The analyst's view of an investigation.

Everything on the screen is read back from TigerGraph, not from the answer files: the case
vertex, its event trail, the transactions and cards it touched, and the closed cases it
drew on. That is the point of writing cases to the graph — an analyst opens the case the
same way the next investigation retrieves it.

Four things the round asks to be visible are each given their own place: how the case
progressed, what evidence it rests on, how certain the agent is, and what it recommends
with the approval each action needs.

Run with:  uv run streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from fraudtrail.answer.schema import Answer
from fraudtrail.config import load_settings
from fraudtrail.graph.client import GraphClient
from fraudtrail.graph.memory import GraphMemory
from fraudtrail.investigate import constants as ic
from fraudtrail.policy.actions import Route
from fraudtrail.wording import counted

REPO = Path(__file__).resolve().parent.parent
CASES_DIR = REPO / "cases"

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

ROUTE_HELP = {
    Route.AUTO: "the agent may do this without asking",
    Route.L1: "a fraud analyst must approve",
    Route.L2: "a fraud manager must approve",
}

VERDICT_COLOUR = {"fraud": "#b3261e", "legitimate": "#146c2e", "uncertain": "#8a6100"}

VIEWS = ["Case", "Queue", "Rings"]


@st.cache_resource
def graph_memory() -> GraphMemory:
    return GraphMemory(GraphClient.from_env())


@st.cache_resource
def graph_client() -> GraphClient:
    return GraphClient.from_env()


@st.cache_data
def load_answers() -> dict[str, Answer]:
    answers: dict[str, Answer] = {}
    for path in sorted(CASES_DIR.glob("HHG-*.json")):
        answers[path.stem] = Answer.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return answers


@st.cache_data(show_spinner="Reading the case from the graph…")
def read_case(case_id: str) -> dict[str, Any]:
    return dict(graph_memory().read(case_id))


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


def money(value: float) -> str:
    return f"${value:,.2f}"


def md(text: str) -> str:
    """Markdown reads a pair of dollar signs as mathematics, and every amount here has one."""
    return text.replace("$", r"\$")


def verdict_badge(answer: Answer) -> str:
    colour = VERDICT_COLOUR.get(answer.case.verdict.value, "#444")
    return (
        f"<span style='background:{colour};color:#fff;padding:2px 10px;"
        f"border-radius:10px;font-weight:600'>{answer.case.verdict.value.upper()}</span>"
    )


def show_header(answer: Answer) -> None:
    st.markdown(
        f"### {answer.case_id} &nbsp; {verdict_badge(answer)}",
        unsafe_allow_html=True,
    )
    left, middle, right, far = st.columns(4)
    left.metric("Fraud probability", f"{answer.case.fraud_probability:.2f}")
    middle.metric("Exposure", money(answer.case.exposure_usd))
    right.metric("Pattern", answer.case.pattern.value.replace("_", " "))
    far.metric("Evidence queries", answer.tool_calls)
    st.write(md(answer.case.summary))
    if answer.case.pattern_description:
        st.info(md(f"**Undocumented pattern.** {answer.case.pattern_description}"))


def show_actions(answer: Answer) -> None:
    st.subheader("Recommended actions")
    st.caption(
        "The agent recommends; it executes nothing that needs a human. "
        "Each action carries the approval its route requires."
    )
    before, after = st.columns(2)
    before.markdown("**Before more evidence was requested**")
    for item in answer.next_best_actions.initial:
        before.markdown(
            f"- `{item.action.value}` — **{item.route.value}** "
            f"<span style='color:#666'>({ROUTE_HELP.get(item.route, '')})</span><br>"
            f"<span style='color:#666;font-size:0.9em'>{md(item.reason)}</span>",
            unsafe_allow_html=True,
        )
    after.markdown("**After the response came back**")
    for item in answer.next_best_actions.final:
        after.markdown(
            f"- `{item.action.value}` — **{item.route.value}** "
            f"<span style='color:#666'>({ROUTE_HELP.get(item.route, '')})</span><br>"
            f"<span style='color:#666;font-size:0.9em'>{md(item.reason)}</span>",
            unsafe_allow_html=True,
        )
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
    if answer.case.connected_card_ids:
        st.markdown(
            f"**Connected cards ({len(answer.case.connected_card_ids)}):** "
            + ", ".join(f"`{c}`" for c in answer.case.connected_card_ids[:12])
            + ("…" if len(answer.case.connected_card_ids) > 12 else "")
        )
    if answer.case.connected_device_profiles:
        for profile in answer.case.connected_device_profiles[:3]:
            st.code(profile, language=None)
    if answer.case.similar_prior_cases:
        st.markdown(
            "**Prior cases retrieved from memory:** "
            + ", ".join(f"`{c}`" for c in answer.case.similar_prior_cases)
        )


def show_progression(case_id: str) -> None:
    st.subheader("Case progression")
    st.caption("Read back from the graph: every step this investigation took, in order.")
    stored = read_case(case_id)
    events = stored.get("events")
    if not isinstance(events, list) or not events:
        st.warning("This case is not in the graph yet. Run scripts/run_agent.py.")
        return
    ordered = sorted(events, key=lambda e: int(e.get("attributes", {}).get("step", 0)))
    for event in ordered:
        a = event.get("attributes", {})
        st.markdown(
            f"**{a.get('step')}. {str(a.get('kind', '')).replace('_', ' ')}** — "
            f"<span style='color:#555'>{md(str(a.get('detail', '')))}</span>",
            unsafe_allow_html=True,
        )


def show_report(answer: Answer) -> None:
    st.subheader("Suspicious activity report")
    if not answer.sar.file:
        st.success(md(f"No report required. {answer.sar.reason}"))
        return
    st.error(md(f"**Report required.** {answer.sar.reason}"))
    st.metric("Total amount", money(answer.sar.total_amount_usd))
    st.markdown(f"**Activity dates:** {' to '.join(answer.sar.activity_dates)}")
    st.markdown(f"**Subjects:** {', '.join(answer.sar.subjects)}")
    st.markdown("**Narrative**")
    st.write(md(answer.sar.narrative))


def show_rings() -> None:
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
    rings = find_rings(min_cards)
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
    st.set_page_config(page_title="FraudTrail", page_icon="🔎", layout="wide")
    settings = load_settings()
    answers = load_answers()

    st.sidebar.title("FraudTrail")
    st.sidebar.caption(f"graph: {settings.tigergraph.graph}")
    # ?view=Queue or ?case=HHG-014 opens that view directly, so a case can be shared as a link.
    requested_view = st.query_params.get("view", VIEWS[0])
    view = st.sidebar.radio(
        "View", VIEWS, index=VIEWS.index(requested_view) if requested_view in VIEWS else 0
    )

    if view == "Queue":
        st.title("Approval queue")
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
        return

    if view == "Rings":
        st.title("Shared-device rings")
        show_rings()
        return

    case_ids = list(answers)
    requested_case = st.query_params.get("case", "")
    case_id = st.sidebar.selectbox(
        "Case",
        case_ids,
        index=case_ids.index(requested_case) if requested_case in case_ids else 0,
        format_func=lambda c: f"{c} — {answers[c].case.verdict.value}",
    )
    answer = answers[case_id]
    st.sidebar.metric("Written to the graph", "yes" if answer.case.written_to_graph else "no")
    st.sidebar.metric("Latency", f"{answer.latency_s:.2f}s")
    st.sidebar.metric("Model tokens", f"{answer.tokens:,}")

    st.title("Fraud investigation")
    show_header(answer)
    tabs = st.tabs(["Progression", "Evidence", "Uncertainty", "Actions", "Report"])
    with tabs[0]:
        show_progression(answer.case.graph_case_id or f"CASE-{case_id}")
    with tabs[1]:
        show_evidence(answer)
    with tabs[2]:
        show_uncertainty(answer)
    with tabs[3]:
        show_actions(answer)
    with tabs[4]:
        show_report(answer)
    st.caption(f"Rendered {datetime.now():%Y-%m-%d %H:%M}")


main()
