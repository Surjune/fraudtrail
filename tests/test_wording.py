from fraudtrail.wording import counted, noun, was_or_were


def test_one_of_a_thing_is_singular() -> None:
    assert counted(1, "card") == "1 card"
    assert was_or_were(1) == "was"


def test_several_and_none_are_plural() -> None:
    assert counted(3, "earlier alert") == "3 earlier alerts"
    assert counted(0, "transaction") == "0 transactions"
    assert was_or_were(2) == "were"


def test_an_irregular_plural_is_used_when_given() -> None:
    assert noun(2, "analysis", "analyses") == "analyses"
    assert noun(1, "analysis", "analyses") == "analysis"
