"""SQL fragments that define derived identifiers, shared by profiling and graph export.

Both rules were verified against the dataset's own labels (see docs/data_findings.md).
"""

from __future__ import annotations


def card_keys_sql(source: str) -> str:
    """One row per card: customer_id, card4, card6, card_id.

    Cards are the distinct (card4, card6) pairs a customer uses, numbered K1, K2, ...
    in ascending order with nulls first. This reproduces all 1,917 card IDs named in the
    closed cases and the case pack.
    """
    return (
        "SELECT customer_id, card4, card6, customer_id || '-K' || row_number() OVER ("
        "PARTITION BY customer_id ORDER BY card4 ASC NULLS FIRST, card6 ASC NULLS FIRST"
        f") AS card_id FROM (SELECT DISTINCT customer_id, card4, card6 FROM {source})"
    )


def card_join_condition(txn_alias: str, keys_alias: str) -> str:
    """Join a transaction to its card; card4/card6 may be null."""
    return (
        f"{keys_alias}.customer_id = {txn_alias}.customer_id "
        f"AND {keys_alias}.card4 IS NOT DISTINCT FROM {txn_alias}.card4 "
        f"AND {keys_alias}.card6 IS NOT DISTINCT FROM {txn_alias}.card6"
    )


def device_profile_sql(identity_alias: str) -> str:
    """Device profile key: DeviceInfo | OS | browser | screen, the README's format.

    Missing parts stay as empty strings so every key has four positions.
    """
    a = identity_alias
    return (
        f"coalesce({a}.DeviceInfo, '') || ' | ' || coalesce({a}.id_30, '') || ' | ' "
        f"|| coalesce({a}.id_31, '') || ' | ' || coalesce({a}.id_33, '')"
    )
