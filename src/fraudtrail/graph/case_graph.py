"""One investigation's neighbourhood in the graph, shaped for drawing.

The `case_graph` query returns the case and the evidence around it as vertex sets and id
maps. `draw` turns that into nodes and edges. Two kinds of edge are kept apart:

- evidence: stored in the graph (a card made a transaction, a transaction came from a
  device), or following directly from it (a linked card made transactions from the case's
  device);
- the case's own links: what the agent wrote when it closed the case (the card it was
  opened on, the closed cases it drew on, a card it connected to the case).

A case link to a transaction, device or connected card is drawn only when the evidence
gives no path to it, so the ring behind a case reads as cards around a device rather than
as a star around the case.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

# Longer labels are cut; the node's details keep the full value.
MAX_LABEL = 24

# The relation drawn from a device to a linked card: the card made transactions from it.
USED_DEVICE = "USED_DEVICE"


class Kind(StrEnum):
    CASE = "case"
    CARD = "card"
    OWNER = "owner"
    OWNER_CARD = "owner_card"
    LINKED_CARD = "linked_card"
    FLAGGED = "flagged"
    TRANSACTION = "transaction"
    DEVICE = "device"
    REGION = "region"
    CLOSED_CASE = "closed_case"


@dataclass(frozen=True)
class Node:
    id: str
    kind: Kind
    label: str
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    case_link: bool = False


@dataclass(frozen=True)
class CaseGraph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    def count(self, *kinds: Kind) -> int:
        return sum(1 for node in self.nodes if node.kind in kinds)

    def for_drawing(self) -> dict[str, list[dict[str, object]]]:
        """Nodes and edges in the shape the network drawing takes."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "group": node.kind.value,
                    "details": [list(row) for row in node.details],
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "from": edge.source,
                    "to": edge.target,
                    "relation": edge.relation,
                    "caseLink": edge.case_link,
                }
                for edge in self.edges
            ],
        }


# The attributes a node's details show, by vertex type, with the name each is shown under.
DETAILS: dict[str, tuple[tuple[str, str], ...]] = {
    "InvestigationCase": (
        ("verdict", "Verdict"),
        ("fraud_probability", "Fraud probability"),
        ("pattern", "Pattern"),
        ("exposure_usd", "Exposure"),
        ("status", "Status"),
        ("report_filed", "Report filed"),
    ),
    "Card": (
        ("network", "Network"),
        ("card_type", "Type"),
        ("txn_count", "Transactions"),
        ("first_seen", "First seen"),
        ("last_seen", "Last seen"),
    ),
    "Transaction": (
        ("ts", "Time"),
        ("amount", "Amount"),
        ("channel", "Channel"),
        ("product_cd", "Product"),
        ("device_status", "Device"),
        ("proxy_flag", "Proxy"),
    ),
    "DeviceProfile": (
        ("device_type", "Type"),
        ("device_info", "Device"),
        ("os", "OS"),
        ("browser", "Browser"),
        ("screen", "Screen"),
    ),
    "ClosedCase": (
        ("outcome", "Outcome"),
        ("pattern", "Pattern"),
        ("opened_at", "Opened"),
        ("exposure_usd", "Exposure"),
        ("n_txns", "Transactions"),
        ("report_filed", "Report filed"),
    ),
}

MONEY_ATTRIBUTES = frozenset({"amount", "exposure_usd"})
WORD_ATTRIBUTES = frozenset({"verdict", "pattern", "status", "outcome", "channel"})

# The link the case wrote to each kind of node, drawn when the evidence gives no path to it.
CASE_LINKS = {
    Kind.DEVICE: "INV_DEVICE",
    Kind.FLAGGED: "INV_FLAGGED",
    Kind.TRANSACTION: "INV_AFFECTS",
    Kind.LINKED_CARD: "INV_CONNECTED_CARD",
}


def _cut(text: str) -> str:
    return text if len(text) <= MAX_LABEL else text[: MAX_LABEL - 1] + "…"


def _shown(name: str, value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if name in MONEY_ATTRIBUTES and isinstance(value, int | float):
        return f"${value:,.2f}"
    if name == "fraud_probability" and isinstance(value, int | float):
        return f"{value:.2f}"
    text = str(value)
    return text.replace("_", " ") if name in WORD_ATTRIBUTES else text


def _attributes(vertex: Mapping[str, object]) -> dict[str, object]:
    raw = vertex.get("attributes")
    if not isinstance(raw, Mapping):
        return {}
    # A printed vertex also carries the query's own accumulators; they are not the vertex's.
    return {str(k): v for k, v in raw.items() if not str(k).startswith("@")}


def _label(v_type: str, v_id: str, attributes: Mapping[str, object]) -> str:
    if v_type == "InvestigationCase":
        return str(attributes.get("exam_case_id") or v_id)
    if v_type == "Transaction":
        amount = attributes.get("amount")
        return f"{v_id}\n${amount:,.2f}" if isinstance(amount, int | float) else v_id
    if v_type == "DeviceProfile":
        return _cut(str(attributes.get("device_info") or attributes.get("os") or v_id))
    if v_type == "BillingRegion":
        return f"region {v_id}"
    return _cut(v_id)


def _node(vertex: Mapping[str, object], kind: Kind) -> Node:
    v_type = str(vertex.get("v_type", ""))
    v_id = str(vertex.get("v_id", ""))
    attributes = _attributes(vertex)
    details = tuple(
        (shown_as, _shown(name, attributes[name]))
        for name, shown_as in DETAILS.get(v_type, ())
        if attributes.get(name) not in (None, "")
    )
    return Node(
        f"{v_type}:{v_id}", kind, _label(v_type, v_id, attributes), (("Id", v_id), *details)
    )


def _vertices(neighbourhood: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    block = neighbourhood.get(key)
    if not isinstance(block, list):
        return []
    return [v for v in block if isinstance(v, Mapping)]


def _id_map(neighbourhood: Mapping[str, object], key: str) -> dict[str, list[str]]:
    """An id map from the query, each value as a list whether it printed one id or a set."""
    block = neighbourhood.get(key)
    if not isinstance(block, Mapping):
        return {}
    ids: dict[str, list[str]] = {}
    for k, v in block.items():
        ids[str(k)] = [str(x) for x in v] if isinstance(v, list) else [str(v)]
    return ids


class _Drawing:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._pairs: set[frozenset[str]] = set()

    def add(self, vertex: Mapping[str, object], kind: Kind) -> str:
        # A vertex in two sets (a linked card the cardholder also owns) keeps its first role.
        node = _node(vertex, kind)
        self.nodes.setdefault(node.id, node)
        return node.id

    def link(self, source: str, target: str, relation: str, *, case_link: bool = False) -> None:
        """One edge per pair of nodes, and only between nodes that are drawn."""
        pair = frozenset((source, target))
        if source not in self.nodes or target not in self.nodes or len(pair) == 1:
            return
        if pair not in self._pairs:
            self._pairs.add(pair)
            self.edges.append(Edge(source, target, relation, case_link))

    def reached_from(self, start: str) -> set[str]:
        neighbours: dict[str, set[str]] = {node_id: set() for node_id in self.nodes}
        for edge in self.edges:
            neighbours[edge.source].add(edge.target)
            neighbours[edge.target].add(edge.source)
        reached, frontier = {start}, [start]
        while frontier:
            for nxt in neighbours[frontier.pop()] - reached:
                reached.add(nxt)
                frontier.append(nxt)
        return reached


def draw(neighbourhood: Mapping[str, object]) -> CaseGraph:
    """The case and the evidence around it as nodes and edges; empty if the case is absent."""
    cases = _vertices(neighbourhood, "investigation")
    if not cases:
        return CaseGraph((), ())
    drawing = _Drawing()
    case = drawing.add(cases[0], Kind.CASE)
    for key, kind in (
        ("card", Kind.CARD),
        ("flagged", Kind.FLAGGED),
        ("affected", Kind.TRANSACTION),
        ("owner", Kind.OWNER),
        ("owner_cards", Kind.OWNER_CARD),
        ("linked_cards", Kind.LINKED_CARD),
        ("devices", Kind.DEVICE),
        ("regions", Kind.REGION),
        ("prior_cases", Kind.CLOSED_CASE),
    ):
        for vertex in _vertices(neighbourhood, key):
            drawing.add(vertex, kind)

    for card in _vertices(neighbourhood, "card"):
        drawing.link(case, f"Card:{card.get('v_id')}", "INV_ON_CARD", case_link=True)
    for owner in _vertices(neighbourhood, "owner"):
        for card in _vertices(neighbourhood, "card") + _vertices(neighbourhood, "owner_cards"):
            drawing.link(f"Customer:{owner.get('v_id')}", f"Card:{card.get('v_id')}", "OWNS")
    for txn, cards in _id_map(neighbourhood, "card_of").items():
        for card_id in cards:
            drawing.link(f"Card:{card_id}", f"Transaction:{txn}", "MADE")
    for txn, devices in _id_map(neighbourhood, "device_of").items():
        for device in devices:
            drawing.link(f"Transaction:{txn}", f"DeviceProfile:{device}", "FROM_DEVICE")
    for txn, regions in _id_map(neighbourhood, "region_of").items():
        for region in regions:
            drawing.link(f"Transaction:{txn}", f"BillingRegion:{region}", "BILLED_IN")
    for device, cards in _id_map(neighbourhood, "linked_on_device").items():
        for card_id in cards:
            drawing.link(f"DeviceProfile:{device}", f"Card:{card_id}", USED_DEVICE)
    closed_case_card = _id_map(neighbourhood, "closed_case_card")
    for prior in _vertices(neighbourhood, "prior_cases"):
        prior_id = f"ClosedCase:{prior.get('v_id')}"
        drawing.link(case, prior_id, "INV_SIMILAR_CLOSED", case_link=True)
        for card_id in closed_case_card.get(str(prior.get("v_id")), []):
            drawing.link(prior_id, f"Card:{card_id}", "ON_CARD")

    # Whatever the evidence leaves unreached hangs from the link the case wrote to it.
    for kind, relation in CASE_LINKS.items():
        for node in [n for n in drawing.nodes.values() if n.kind is kind]:
            if node.id not in drawing.reached_from(case):
                drawing.link(case, node.id, relation, case_link=True)
    return CaseGraph(tuple(drawing.nodes.values()), tuple(drawing.edges))
