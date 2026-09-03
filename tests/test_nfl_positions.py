"""Position normalization + target selection."""
import pytest

from nfl.config import (
    normalize_position, POSITION_TARGETS, SUPPORTED_POSITIONS, all_targets,
    position_support_message, COLUMN_ALIASES,
)


@pytest.mark.parametrize("raw,expected", [
    ("QB", "QB"), ("RB", "RB"), ("HB", "RB"), ("FB", "RB"),
    ("WR", "WR"), ("SE", "WR"), ("FL", "WR"), ("TE", "TE"),
    ("K", "K"), ("PK", "K"), ("qb", "QB"),
])
def test_normalize_supported(raw, expected):
    assert normalize_position(raw) == expected


@pytest.mark.parametrize("raw", ["P", "LS", "OL", "C", "G", "T", "DE", "CB", "S", "LB", "DST"])
def test_unsupported_positions_return_none(raw):
    assert normalize_position(raw) is None


def test_never_guesses_from_unknown_label():
    assert normalize_position("ATHLETE") is None
    assert normalize_position(None) is None


def test_position_group_fallback():
    # position missing/odd but group is canonical
    assert normalize_position("", "RUNNING_BACK") == "RB"
    assert normalize_position(None, "WIDE_RECEIVER") == "WR"


def test_specialist_group_not_guessed():
    # SPECIALIST group is ambiguous (K/P/LS) -> None unless position says K
    assert normalize_position("P", "SPEC") is None
    assert normalize_position("K", "SPEC") == "K"


def test_support_message_mentions_supported():
    msg = position_support_message("P")
    assert "Punter" in msg or "not supported" in msg


def test_every_position_has_targets():
    for pos in SUPPORTED_POSITIONS:
        assert pos in POSITION_TARGETS
        assert len(POSITION_TARGETS[pos]) > 0


def test_target_selection_matches_position():
    assert "passing_yards" in POSITION_TARGETS["QB"]
    assert "passing_yards" not in POSITION_TARGETS["WR"]  # WR shows no passing
    assert "carries" in POSITION_TARGETS["RB"]
    assert "kicking_points" in POSITION_TARGETS["K"]
    assert "receiving_yards" in POSITION_TARGETS["TE"]


def test_all_targets_covers_each_position():
    at = set(all_targets())
    for pos in SUPPORTED_POSITIONS:
        assert set(POSITION_TARGETS[pos]).issubset(at)


def test_column_aliases_documented_for_targets():
    for target in all_targets():
        assert target in COLUMN_ALIASES  # every canonical target has a mapping entry
