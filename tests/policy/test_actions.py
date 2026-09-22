import pytest

from fraudtrail.policy.actions import EXECUTION_ORDER, Action, Route, route_for


def test_every_action_has_a_route_and_a_place_in_the_order() -> None:
    assert set(EXECUTION_ORDER) == set(Action)
    assert len(EXECUTION_ORDER) == len(set(EXECUTION_ORDER))
    for action in Action:
        route_for(action, exposure_usd=0.0)


@pytest.mark.parametrize(
    ("action", "exposure", "route"),
    [
        (Action.VERIFY_WITH_CUSTOMER, 10_000.0, Route.AUTO),
        (Action.CREATE_CASE, 10_000.0, Route.AUTO),
        (Action.ESCALATE_TO_ANALYST, 10_000.0, Route.AUTO),
        (Action.DECLINE_TRANSACTION, 0.0, Route.L1),
        (Action.BLOCK_CARD, 2_500.0, Route.L1),
        (Action.BLOCK_CARD, 2_500.01, Route.L2),
        (Action.BLOCK_ALL_CARDS, 0.0, Route.L2),
        (Action.FILE_REPORT, 0.0, Route.L2),
    ],
)
def test_routes_follow_section_2(action: Action, exposure: float, route: Route) -> None:
    assert route_for(action, exposure) is route
