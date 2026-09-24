from datetime import datetime

from fraudtrail.evidence.models import in_order_of_first_use


def test_cards_are_listed_in_the_order_they_first_used_the_device() -> None:
    first_used = {
        "C3-K1": datetime(2016, 11, 20, 9, 0),
        "C1-K1": datetime(2016, 11, 22, 9, 0),
        "C2-K1": datetime(2016, 11, 14, 9, 0),
    }
    assert in_order_of_first_use(first_used) == ("C2-K1", "C3-K1", "C1-K1")


def test_a_tie_is_broken_by_id_so_every_run_agrees() -> None:
    same = datetime(2016, 11, 14, 9, 0)
    assert in_order_of_first_use({"C9-K1": same, "C1-K1": same}) == ("C1-K1", "C9-K1")
    assert in_order_of_first_use({"C1-K1": same, "C9-K1": same}) == ("C1-K1", "C9-K1")
