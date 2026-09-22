"""Action identifiers, approval routing and execution order (Fraud Policy sections 1 and 2)."""

from __future__ import annotations

from enum import StrEnum

from fraudtrail.policy import constants as c


class Action(StrEnum):
    ALLOW_TRANSACTION = "ALLOW_TRANSACTION"
    DECLINE_TRANSACTION = "DECLINE_TRANSACTION"
    MONITOR_CARD = "MONITOR_CARD"
    MONITOR_CONNECTED_CARDS = "MONITOR_CONNECTED_CARDS"
    WARN_CUSTOMER = "WARN_CUSTOMER"
    VERIFY_WITH_CUSTOMER = "VERIFY_WITH_CUSTOMER"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    BLOCK_CARD = "BLOCK_CARD"
    BLOCK_ALL_CARDS = "BLOCK_ALL_CARDS"
    GENERATE_REPORT = "GENERATE_REPORT"
    CREATE_CASE = "CREATE_CASE"
    FILE_REPORT = "FILE_REPORT"
    ESCALATE_TO_ANALYST = "ESCALATE_TO_ANALYST"
    CLOSE_NO_FRAUD = "CLOSE_NO_FRAUD"


class Route(StrEnum):
    AUTO = "auto"
    L1 = "L1"
    L2 = "L2"


AUTO_ACTIONS: frozenset[Action] = frozenset(
    {
        Action.ALLOW_TRANSACTION,
        Action.MONITOR_CARD,
        Action.MONITOR_CONNECTED_CARDS,
        Action.WARN_CUSTOMER,
        Action.VERIFY_WITH_CUSTOMER,
        Action.STEP_UP_AUTH,
        Action.GENERATE_REPORT,
        Action.CREATE_CASE,
        Action.ESCALATE_TO_ANALYST,
        Action.CLOSE_NO_FRAUD,
    }
)

BLOCKING_ACTIONS: frozenset[Action] = frozenset({Action.BLOCK_CARD, Action.BLOCK_ALL_CARDS})


def route_for(action: Action, exposure_usd: float) -> Route:
    """Approval route for an action at a given exposure (section 2)."""
    if action in AUTO_ACTIONS:
        return Route.AUTO
    if action is Action.DECLINE_TRANSACTION:
        return Route.L1
    if action is Action.BLOCK_CARD:
        if exposure_usd <= c.BLOCK_CARD_L1_MAX_EXPOSURE_USD:
            return Route.L1
        return Route.L2
    if action in (Action.BLOCK_ALL_CARDS, Action.FILE_REPORT):
        return Route.L2
    raise ValueError(f"No approval route defined for {action}")


# Section 1: "Order them by what happens first." Stop the money first, then gather
# evidence, then write the record and the filing, then monitor and hand off, then close.
EXECUTION_ORDER: tuple[Action, ...] = (
    Action.DECLINE_TRANSACTION,
    Action.BLOCK_ALL_CARDS,
    Action.BLOCK_CARD,
    Action.STEP_UP_AUTH,
    Action.VERIFY_WITH_CUSTOMER,
    Action.CREATE_CASE,
    Action.FILE_REPORT,
    Action.MONITOR_CARD,
    Action.MONITOR_CONNECTED_CARDS,
    Action.WARN_CUSTOMER,
    Action.ESCALATE_TO_ANALYST,
    Action.GENERATE_REPORT,
    Action.ALLOW_TRANSACTION,
    Action.CLOSE_NO_FRAUD,
)
