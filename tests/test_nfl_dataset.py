"""build_dataset behaviour: duplicate names, traded players, bye weeks,
canonical IDs, and graceful handling of missing optional datasets."""
import polars as pl
import pandas as pd

from nfl.scraping.collect import DataBundle
from nfl.features.build_dataset import build_dataset


def _player_stats():
    # Two DIFFERENT players share the name "John Smith" (distinct ids).
    # Player p_trade is traded: team changes from AAA to BBB across weeks.
    rows = []
    def rec(pid, name, pos, season, week, team, opp, game, **extra):
        base = dict(player_id=pid, player_name=name, player_display_name=name,
                    position=pos, position_group=pos, season=season, week=week,
                    season_type="REG", game_id=game, team=team, opponent_team=opp,
                    receiving_yards=50.0, receptions=4, targets=6, fg_made=0,
                    pat_made=0, fumbles_lost_total=0)
        base.update(extra)
        rows.append(base)

    # duplicate-name WRs
    rec("p_a", "John Smith", "WR", 2023, 1, "AAA", "BBB", "2023_01_AAA_BBB")
    rec("p_b", "John Smith", "WR", 2023, 1, "CCC", "DDD", "2023_01_CCC_DDD")
    # traded player across weeks (bye at week 3 -> missing week 3)
    rec("p_trade", "Trade Guy", "RB", 2023, 1, "AAA", "BBB", "2023_01_AAA_BBB",
        rushing_yards=40.0, carries=10)
    rec("p_trade", "Trade Guy", "RB", 2023, 2, "AAA", "CCC", "2023_02_AAA_CCC",
        rushing_yards=55.0, carries=12)
    rec("p_trade", "Trade Guy", "RB", 2023, 4, "BBB", "DDD", "2023_04_BBB_DDD",
        rushing_yards=33.0, carries=9)
    return pl.DataFrame(rows)


def _players():
    return pl.DataFrame([
        dict(gsis_id="p_a", display_name="John Smith", position="WR",
             position_group="WR", years_of_experience=3, pfr_id="SmitJo00", headshot="u1"),
        dict(gsis_id="p_b", display_name="John Smith", position="WR",
             position_group="WR", years_of_experience=5, pfr_id="SmitJo01", headshot="u2"),
        dict(gsis_id="p_trade", display_name="Trade Guy", position="RB",
             position_group="RB", years_of_experience=2, pfr_id="GuyTr00", headshot="u3"),
    ])


def _schedules():
    def g(game, season, week, home, away, **extra):
        # spread_line +3 => home favored by 3 (nflverse convention).
        base = dict(game_id=game, season=season, week=week, home_team=home,
                    away_team=away, gameday=f"2023-09-{week:02d}", gametime="13:00",
                    total_line=45.0, spread_line=3.0, home_rest=7, away_rest=7,
                    roof="outdoors", surface="grass", temp=70, wind=5, div_game=0)
        base.update(extra)
        return base
    return pl.DataFrame([
        g("2023_01_AAA_BBB", 2023, 1, "AAA", "BBB"),
        g("2023_01_CCC_DDD", 2023, 1, "CCC", "DDD"),
        g("2023_02_AAA_CCC", 2023, 2, "AAA", "CCC"),
        g("2023_04_BBB_DDD", 2023, 4, "BBB", "DDD"),
    ])


def _bundle(**opt):
    return DataBundle(
        seasons=[2023], player_stats=_player_stats(), players=_players(),
        schedules=_schedules(),
        injuries=opt.get("injuries"), snap_counts=opt.get("snap_counts"),
        depth_charts=opt.get("depth_charts"), rosters_weekly=opt.get("rosters_weekly"),
        availability={"optional": {}},
    )


def test_duplicate_names_kept_distinct_by_id():
    ds = build_dataset(_bundle())
    smiths = ds[ds["player_name"] == "John Smith"]
    assert set(smiths["gsis_id"]) == {"p_a", "p_b"}   # two distinct canonical ids


def test_canonical_id_column_present():
    ds = build_dataset(_bundle())
    assert "gsis_id" in ds.columns
    # every row uniquely identified by (gsis_id, game_id)
    assert not ds.duplicated(["gsis_id", "game_id"]).any()


def test_traded_player_team_tracked_per_week():
    ds = build_dataset(_bundle())
    tg = ds[ds["gsis_id"] == "p_trade"].sort_values("week")
    assert tg[tg["week"] == 1]["team"].iloc[0] == "AAA"
    assert tg[tg["week"] == 4]["team"].iloc[0] == "BBB"   # traded to BBB


def test_bye_week_gap_present_in_data():
    ds = build_dataset(_bundle())
    weeks = sorted(ds[ds["gsis_id"] == "p_trade"]["week"].tolist())
    assert weeks == [1, 2, 4]   # week 3 absent (bye/inactive)


def test_missing_optional_datasets_do_not_crash():
    # No injuries/snaps/depth/rosters at all.
    ds = build_dataset(_bundle())
    assert (ds["missing_injury"] == 1).all()
    assert (ds["missing_snap_share"] == 1).all()
    assert (ds["missing_depth"] == 1).all()


def test_home_away_and_implied_total_computed():
    ds = build_dataset(_bundle())
    # AAA is home vs BBB with spread_line -3 (home favored by 3) -> team_spread -3
    row = ds[(ds["gsis_id"] == "p_a")].iloc[0]
    assert row["home_away"] == 1
    # implied = total/2 - team_spread/2 = 22.5 + 1.5 = 24 for favored home team
    assert abs(row["implied_team_total"] - 24.0) < 1e-6


def test_only_supported_positions_retained():
    b = _bundle()
    # inject a punter row that must be filtered out
    extra = b.player_stats.to_pandas()
    extra = pd.concat([extra, pd.DataFrame([dict(
        player_id="punter", player_name="Punter Guy", player_display_name="Punter Guy",
        position="P", position_group="SPEC", season=2023, week=1, season_type="REG",
        game_id="2023_01_AAA_BBB", team="AAA", opponent_team="BBB")])], ignore_index=True)
    b.player_stats = pl.from_pandas(extra)
    ds = build_dataset(b)
    assert "punter" not in set(ds["gsis_id"])
