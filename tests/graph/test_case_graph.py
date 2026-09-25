from __future__ import annotations

from fraudtrail.graph.case_graph import MAX_LABEL, USED_DEVICE, CaseGraph, Kind, draw

DEVICE = "SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for android | 1920x1080"


def vertex(v_type: str, v_id: str, **attributes: object) -> dict[str, object]:
    # Printed vertices carry the query's accumulators too; draw must leave them out.
    return {"v_id": v_id, "v_type": v_type, "attributes": {**attributes, "@on_case": False}}


def ring_case() -> dict[str, object]:
    """HHG-014 in small: a flagged purchase on a device two other cards also used."""
    return {
        "investigation": [
            vertex(
                "InvestigationCase",
                "CASE-HHG-014",
                exam_case_id="HHG-014",
                verdict="fraud",
                fraud_probability=0.96,
                exposure_usd=187.33,
                report_filed=True,
                summary="An analyst requested a review.",
            )
        ],
        "card": [vertex("Card", "C13487-K1", network="mastercard", card_type="debit")],
        "owner": [vertex("Customer", "C13487")],
        "owner_cards": [vertex("Card", "C13487-K1"), vertex("Card", "C13487-K2")],
        "flagged": [
            vertex("Transaction", "3478561", amount=74.96, proxy_flag="IP_PROXY:ANONYMOUS")
        ],
        "affected": [
            vertex("Transaction", "3478561", amount=74.96),
            vertex("Transaction", "3460634", amount=112.37),
        ],
        "devices": [vertex("DeviceProfile", DEVICE, device_info="SM-G935F Build/NRD90M")],
        "regions": [vertex("BillingRegion", "191")],
        "linked_cards": [
            vertex("Card", "C05448-K2"),
            vertex("Card", "C01289-K1"),
            vertex("Card", "C09999-K1"),
        ],
        "prior_cases": [vertex("ClosedCase", "CC-2985", outcome="confirmed_fraud")],
        "card_of": {"3478561": "C13487-K1", "3460634": "C13487-K1"},
        "device_of": {"3478561": DEVICE, "3460634": DEVICE},
        "region_of": {"3478561": "191", "3460634": "191"},
        "linked_on_device": {DEVICE: ["C05448-K2", "C01289-K1"]},
        "closed_case_card": {"CC-2985": "C06617-K1"},
    }


def edge(graph: CaseGraph, a: str, b: str) -> tuple[str, bool] | None:
    for e in graph.edges:
        if {e.source, e.target} == {a, b}:
            return e.relation, e.case_link
    return None


def test_no_case_draws_nothing() -> None:
    assert draw({}) == CaseGraph((), ())
    assert draw({"investigation": []}).nodes == ()


def test_every_vertex_becomes_one_node_in_its_first_role() -> None:
    graph = draw(ring_case())
    kinds = {n.id: n.kind for n in graph.nodes}
    assert kinds["InvestigationCase:CASE-HHG-014"] is Kind.CASE
    assert kinds["Card:C13487-K1"] is Kind.CARD
    assert kinds["Card:C13487-K2"] is Kind.OWNER_CARD
    # The flagged transaction is also affected; it is drawn once, as flagged.
    assert kinds["Transaction:3478561"] is Kind.FLAGGED
    assert kinds["Transaction:3460634"] is Kind.TRANSACTION
    assert len(graph.nodes) == len(kinds)
    assert graph.count(Kind.FLAGGED, Kind.TRANSACTION) == 2
    assert graph.count(Kind.CARD, Kind.OWNER_CARD, Kind.LINKED_CARD) == 5


def test_evidence_edges_follow_the_graph() -> None:
    graph = draw(ring_case())
    assert edge(graph, "Customer:C13487", "Card:C13487-K2") == ("OWNS", False)
    assert edge(graph, "Card:C13487-K1", "Transaction:3478561") == ("MADE", False)
    assert edge(graph, "Transaction:3478561", f"DeviceProfile:{DEVICE}") == ("FROM_DEVICE", False)
    assert edge(graph, "Transaction:3460634", "BillingRegion:191") == ("BILLED_IN", False)
    assert edge(graph, f"DeviceProfile:{DEVICE}", "Card:C05448-K2") == (USED_DEVICE, False)


def test_the_ring_hangs_from_the_device_not_from_the_case() -> None:
    graph = draw(ring_case())
    case = "InvestigationCase:CASE-HHG-014"
    assert edge(graph, case, "Card:C05448-K2") is None
    assert edge(graph, case, "Transaction:3478561") is None
    assert edge(graph, case, f"DeviceProfile:{DEVICE}") is None


def test_a_card_the_evidence_does_not_reach_gets_the_case_link() -> None:
    graph = draw(ring_case())
    case = "InvestigationCase:CASE-HHG-014"
    assert edge(graph, case, "Card:C09999-K1") == ("INV_CONNECTED_CARD", True)
    assert edge(graph, case, "Card:C13487-K1") == ("INV_ON_CARD", True)
    assert edge(graph, case, "ClosedCase:CC-2985") == ("INV_SIMILAR_CLOSED", True)


def test_a_device_without_a_case_transaction_gets_the_case_link() -> None:
    neighbourhood = ring_case()
    neighbourhood["device_of"] = {}
    neighbourhood["linked_on_device"] = {}
    graph = draw(neighbourhood)
    assert edge(graph, "InvestigationCase:CASE-HHG-014", f"DeviceProfile:{DEVICE}") == (
        "INV_DEVICE",
        True,
    )


def test_a_closed_case_links_to_its_card_only_when_that_card_is_drawn() -> None:
    neighbourhood = ring_case()
    assert edge(draw(neighbourhood), "ClosedCase:CC-2985", "Card:C06617-K1") is None
    neighbourhood["closed_case_card"] = {"CC-2985": "C13487-K2"}
    assert edge(draw(neighbourhood), "ClosedCase:CC-2985", "Card:C13487-K2") == ("ON_CARD", False)


def test_one_edge_per_pair_and_everything_reaches_the_case() -> None:
    graph = draw(ring_case())
    pairs = [frozenset((e.source, e.target)) for e in graph.edges]
    assert len(pairs) == len(set(pairs))
    ids = {n.id for n in graph.nodes}
    assert all(e.source in ids and e.target in ids for e in graph.edges)
    reached, frontier = {"InvestigationCase:CASE-HHG-014"}, ["InvestigationCase:CASE-HHG-014"]
    while frontier:
        here = frontier.pop()
        for e in graph.edges:
            there = e.target if e.source == here else e.source if e.target == here else None
            if there and there not in reached:
                reached.add(there)
                frontier.append(there)
    assert reached == ids


def test_labels_and_details_read_well() -> None:
    nodes = {n.id: n for n in draw(ring_case()).nodes}
    case = nodes["InvestigationCase:CASE-HHG-014"]
    assert case.label == "HHG-014"
    assert ("Fraud probability", "0.96") in case.details
    assert ("Exposure", "$187.33") in case.details
    assert ("Report filed", "yes") in case.details
    # Free text the model wrote is not part of a node.
    assert all("analyst requested" not in value for _, value in case.details)
    assert all(not name.startswith("@") for n in nodes.values() for name, _ in n.details)
    assert nodes["Transaction:3478561"].label == "3478561\n$74.96"
    assert nodes["ClosedCase:CC-2985"].details[1] == ("Outcome", "confirmed fraud")
    device = nodes[f"DeviceProfile:{DEVICE}"]
    assert len(device.label) <= MAX_LABEL
    assert ("Id", DEVICE) in device.details


def test_for_drawing_is_plain_data() -> None:
    drawn = draw(ring_case()).for_drawing()
    first = drawn["nodes"][0]
    assert first["group"] == "case"
    assert first["details"][0] == ["Id", "CASE-HHG-014"]
    assert {"from", "to", "relation", "caseLink"} <= set(drawn["edges"][0])
