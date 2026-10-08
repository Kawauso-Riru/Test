import pandas as pd

from keiba_ai.bet_rules import allocate_daily_budget, recommend_bet


def _race_df(n, surface, distance, scores, umaban=None, popularity=None, race_class="未勝利"):
    umaban = umaban or list(range(1, n + 1))
    return pd.DataFrame({
        "umaban": umaban,
        "horse_name": [f"horse{u}" for u in umaban],
        "score": scores,
        "surface": [surface] * n,
        "distance": [distance] * n,
        "field_size": [n] * n,
        "race_class": [race_class] * n,
        "popularity_numeric": popularity if popularity is not None else [float("nan")] * n,
    })


def test_turf_upset_condition_recommends_wide_3_6():
    # 16-horse field, 1400m turf -> matches the big-field/short-distance rule
    scores = list(range(16, 0, -1))
    race_df = _race_df(16, "芝", 1400, scores)
    rec = recommend_bet(race_df, "r1", "label", "芝")
    assert rec.tier == "S"
    assert rec.bet_type == "ワイド"
    # model rank 3 is umaban 3 (score 14), rank 6 is umaban 6 (score 11)
    assert "3-6番" in rec.target


def test_turf_outside_condition_falls_back_to_fukusho():
    scores = list(range(10, 0, -1))
    race_df = _race_df(10, "芝", 2000, scores)  # long distance, small field
    rec = recommend_bet(race_df, "r2", "label", "芝")
    assert rec.tier == "B"
    assert rec.bet_type == "複勝"


def test_dirt_low_class_and_market_match_is_tier_s():
    scores = [10, 9, 8]
    # umaban 1 is both model's #1 (highest score) and market favorite (popularity 1)
    race_df = _race_df(3, "ダート", 1600, scores, popularity=[1, 2, 3], race_class="未勝利")
    rec = recommend_bet(race_df, "r3", "label", "ダート")
    assert rec.tier == "S"
    assert "1番" in rec.target


def test_dirt_disagreement_pivots_to_market_favorite():
    scores = [10, 9, 8]
    # model's #1 is umaban 1, but market favorite is umaban 2
    race_df = _race_df(3, "ダート", 1600, scores, popularity=[2, 1, 3], race_class="オープン")
    rec = recommend_bet(race_df, "r4", "label", "ダート")
    assert rec.tier == "A"
    assert "2番" in rec.target
    assert "市場1番人気" in rec.target


def test_allocate_daily_budget_prioritizes_higher_tier_and_respects_cap():
    from keiba_ai.bet_rules import BetRecommendation

    recs = [
        BetRecommendation("r1", "r1", "複勝", "1番", "x", "B", 78.6),
        BetRecommendation("r2", "r2", "ワイド", "3-6番", "x", "S", 138.0),
        BetRecommendation("r3", "r3", "複勝", "1番", "x", "A", 86.8),
    ]
    funded, skipped = allocate_daily_budget(recs, daily_budget=4000, unit=100)
    total = sum(stake for _, stake in funded)
    assert total <= 4000
    # S-tier must be funded before lower tiers given the budget cap
    funded_ids = [r.race_id for r, _ in funded]
    assert "r2" in funded_ids
    assert funded_ids[0] == "r2"


def test_allocate_daily_budget_empty_input():
    funded, skipped = allocate_daily_budget([], daily_budget=10000)
    assert funded == []
    assert skipped == []
