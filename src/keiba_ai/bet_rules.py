"""Turn one race's model scores into a concrete bet recommendation (bet
type + horses + a confidence tier), and allocate a fixed daily budget
across a day's worth of these recommendations.

Every rule here is a direct readout of a backtest documented in README.md
("馬券戦略の検証" for dirt, "芝モデルについて > 検証結果" for turf) --
this module intentionally adds no new judgment calls of its own. The tier
assigned to each rule reflects how rigorously that backtest held up
(multi-seed pooling, fresh-seed re-verification), and only `allocate_daily_
budget` uses the tier, to size stakes -- nothing here claims a rule is
profitable beyond what its own backtest showed.

Honesty note baked into the design: only ONE rule in this project has ever
cleared 100% ROI on an independently-verified split (turf, TIER_S below).
Every other rule, including the dirt TIER_A ones, still loses to the JRA
takeout on average -- they're ranked relative to each other, not because
any of them is a genuine positive-EV bet. Treat low-tier picks as
"least-bad", not "profitable".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Confidence tiers, highest first. S = independently re-verified ROI>100%
# (this project's only such result). A = multi-seed-stable but still <100%
# ROI. B = plain, unfiltered default -- "nothing special about this race".
TIER_ORDER = {"S": 3, "A": 2, "B": 1}

LOW_CLASS_RACE_CLASSES = {"未勝利", "1勝クラス"}

# scripts/upset_condition_backtest.py: turf, distance<=1400m, field>=14 ->
# wide(rank3, rank6) pooled ROI 138.0% (n=520) across two independent seed
# sets (114.2% then 183.6%). See README's "🎯 大穴条件...2回連続で黒字化".
TURF_SHORT_DISTANCE_MAX = 1400
TURF_BIG_FIELD_MIN = 14
TURF_CONDITION_WIDE_ROI = 138.0

# scripts/turf_model_roi_backtest.py: turf model, 6-seed pool, n=1796.
TURF_DEFAULT_FUKUSHO_ROI = 82.3

# scripts/robustness_class_confidence_backtest.py + best_segment_backtest.py
# (dirt, 852-1182 holdout races/seed): 複勝 baseline, then stacked filters.
DIRT_DEFAULT_FUKUSHO_ROI = 78.6
DIRT_LOW_CLASS_FUKUSHO_ROI = 83.4
DIRT_MODEL_MATCHES_MARKET_ROI = 86.8
DIRT_LOW_CLASS_AND_MATCH_ROI = 88.3

# scripts/disagreement_rank_backtest.py (dirt, 3-seed pool, n=1958): when
# the model's #1 pick does NOT match the market's actual favorite, the
# model's #1 is the worst option in the race (複勝 76.1%) -- the market's
# real favorite does better (複勝 85.0%).
DIRT_DISAGREEMENT_MODEL_PICK_ROI = 76.1
DIRT_DISAGREEMENT_MARKET_FAVORITE_ROI = 85.0


@dataclass
class BetRecommendation:
    race_id: str
    race_label: str
    bet_type: str
    target: str
    rationale: str
    tier: str
    expected_roi_pct: float


def _market_favorite(race_df: pd.DataFrame) -> pd.Series | None:
    if "popularity_numeric" not in race_df.columns:
        return None
    favorites = race_df[race_df["popularity_numeric"] == 1]
    if favorites.empty:
        return None
    return favorites.iloc[0]


def recommend_turf(race_df: pd.DataFrame, race_id: str, race_label: str) -> BetRecommendation | None:
    ranked = race_df.sort_values("score", ascending=False).reset_index(drop=True)
    if len(ranked) < 6:
        return None

    distance = ranked["distance"].iloc[0]
    field_size = ranked["field_size"].iloc[0] if "field_size" in ranked.columns else len(ranked)
    if pd.notna(distance) and distance <= TURF_SHORT_DISTANCE_MAX and field_size >= TURF_BIG_FIELD_MIN:
        h3, h6 = ranked.iloc[2], ranked.iloc[5]
        return BetRecommendation(
            race_id=race_id,
            race_label=race_label,
            bet_type="ワイド",
            target=f"{int(h3['umaban'])}-{int(h6['umaban'])}番 (指数3位{h3['horse_name']} / 指数6位{h6['horse_name']})",
            rationale=(
                f"芝・距離{int(distance)}m(<=1400m)・出走{int(field_size)}頭(>=14頭)の大穴条件に一致。"
                f"この条件下でのワイド3位-6位は独立した2つのシードセットで黒字化(プールROI{TURF_CONDITION_WIDE_ROI:.1f}%, n=520)"
            ),
            tier="S",
            expected_roi_pct=TURF_CONDITION_WIDE_ROI,
        )

    h1 = ranked.iloc[0]
    return BetRecommendation(
        race_id=race_id,
        race_label=race_label,
        bet_type="複勝",
        target=f"{int(h1['umaban'])}番 ({h1['horse_name']})",
        rationale=(
            "芝は大穴条件に一致せず、検証済みの黒字買い目が無いためデフォルトの複勝"
            f"(6シードプールROI{TURF_DEFAULT_FUKUSHO_ROI:.1f}%, n=1796)。この買い目自体は黒字化していません"
        ),
        tier="B",
        expected_roi_pct=TURF_DEFAULT_FUKUSHO_ROI,
    )


def recommend_dirt(race_df: pd.DataFrame, race_id: str, race_label: str) -> BetRecommendation | None:
    ranked = race_df.sort_values("score", ascending=False).reset_index(drop=True)
    if ranked.empty:
        return None

    h1 = ranked.iloc[0]
    race_class = ranked["race_class"].iloc[0] if "race_class" in ranked.columns else "不明"
    is_low_class = race_class in LOW_CLASS_RACE_CLASSES
    favorite = _market_favorite(ranked)
    model_matches_market = favorite is not None and str(int(favorite["umaban"])) == str(int(h1["umaban"]))

    if favorite is not None and not model_matches_market:
        return BetRecommendation(
            race_id=race_id,
            race_label=race_label,
            bet_type="複勝",
            target=f"{int(favorite['umaban'])}番 ({favorite['horse_name']}, 市場1番人気)",
            rationale=(
                "モデルの1位予想が市場の1番人気と不一致。この場合モデルの1位予想の複勝"
                f"({DIRT_DISAGREEMENT_MODEL_PICK_ROI:.1f}%)より市場の本命の複勝"
                f"({DIRT_DISAGREEMENT_MARKET_FAVORITE_ROI:.1f}%)の方が実績が良い(3シードプールn=1958)"
            ),
            tier="A",
            expected_roi_pct=DIRT_DISAGREEMENT_MARKET_FAVORITE_ROI,
        )

    if model_matches_market and is_low_class:
        return BetRecommendation(
            race_id=race_id,
            race_label=race_label,
            bet_type="複勝",
            target=f"{int(h1['umaban'])}番 ({h1['horse_name']})",
            rationale=(
                f"未勝利/1勝クラス({race_class})かつモデル1位=市場1番人気。このセッションで最も安定した"
                f"実績(複勝ROI{DIRT_LOW_CLASS_AND_MATCH_ROI:.1f}%, 標準偏差2.3pt, n=1947)"
            ),
            tier="S",
            expected_roi_pct=DIRT_LOW_CLASS_AND_MATCH_ROI,
        )

    if model_matches_market:
        return BetRecommendation(
            race_id=race_id,
            race_label=race_label,
            bet_type="複勝",
            target=f"{int(h1['umaban'])}番 ({h1['horse_name']})",
            rationale=f"モデル1位=市場1番人気(複勝ROI{DIRT_MODEL_MATCHES_MARKET_ROI:.1f}%, n=2673)",
            tier="A",
            expected_roi_pct=DIRT_MODEL_MATCHES_MARKET_ROI,
        )

    if is_low_class:
        return BetRecommendation(
            race_id=race_id,
            race_label=race_label,
            bet_type="複勝",
            target=f"{int(h1['umaban'])}番 ({h1['horse_name']})",
            rationale=f"未勝利/1勝クラス({race_class})に限定(複勝ROI{DIRT_LOW_CLASS_FUKUSHO_ROI:.1f}%, 標準偏差1.2pt)",
            tier="A",
            expected_roi_pct=DIRT_LOW_CLASS_FUKUSHO_ROI,
        )

    return BetRecommendation(
        race_id=race_id,
        race_label=race_label,
        bet_type="複勝",
        target=f"{int(h1['umaban'])}番 ({h1['horse_name']})",
        rationale=f"絞り込み条件に合致せず。デフォルトの複勝(全体ROI{DIRT_DEFAULT_FUKUSHO_ROI:.1f}%, n=852)。黒字化していません",
        tier="B",
        expected_roi_pct=DIRT_DEFAULT_FUKUSHO_ROI,
    )


def recommend_bet(race_df: pd.DataFrame, race_id: str, race_label: str, surface: str) -> BetRecommendation | None:
    if surface == "芝":
        return recommend_turf(race_df, race_id, race_label)
    if surface == "ダート":
        return recommend_dirt(race_df, race_id, race_label)
    return None


# Per-race stake if funded at that tier -- intentionally small even at tier
# S, since even the best-verified rule here (S) is a single real-money bet
# type/pair, not a guaranteed win; this just reflects relative confidence.
_BASE_STAKE = {"S": 3000, "A": 1500, "B": 800}


def allocate_daily_budget(
    recommendations: list[BetRecommendation], daily_budget: int = 10000, unit: int = 100
) -> tuple[list[tuple[BetRecommendation, int]], list[BetRecommendation]]:
    """Greedily fund the highest-tier/highest-expected-ROI recommendations
    first until the daily budget is used up. Returns (funded, skipped) --
    `skipped` is not a judgment that those races are bad, just that the
    fixed daily budget ran out before reaching them (see README's note on
    not forcing a fixed stake onto every race regardless of edge)."""
    ranked = sorted(recommendations, key=lambda r: (-TIER_ORDER[r.tier], -r.expected_roi_pct))
    funded: list[tuple[BetRecommendation, int]] = []
    skipped: list[BetRecommendation] = []
    remaining = daily_budget
    for rec in ranked:
        stake = min(_BASE_STAKE[rec.tier], remaining)
        stake = (stake // unit) * unit
        if stake < unit:
            skipped.append(rec)
            continue
        funded.append((rec, stake))
        remaining -= stake
    return funded, skipped
