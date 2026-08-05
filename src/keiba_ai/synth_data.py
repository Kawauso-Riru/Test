"""Synthetic race-result dataset generator.

Produces a dataset with the same schema as parsed netkeiba race-result
pages (one row per horse per race), so the rest of the pipeline
(features -> model -> app) can be built, tested, and demoed without
scraping a live site. Finishing order is driven by a hidden per-horse
"ability" and per-jockey "skill" plus noise, so there is real, learnable
signal for the model to pick up on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PLACES = ["東京", "中山", "阪神", "京都", "中京", "小倉", "新潟", "福島", "札幌", "函館"]
SURFACES = ["芝", "ダート"]
DISTANCES = [1200, 1400, 1600, 1800, 2000, 2400]
TRACK_CONDITIONS = ["良", "稍重", "重", "不良"]
TRACK_CONDITION_PROBS = [0.6, 0.25, 0.1, 0.05]


def generate_synthetic_results(
    n_horses: int = 300,
    n_jockeys: int = 40,
    n_races: int = 600,
    field_size: int = 14,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    horse_ids = [f"h{idx:05d}" for idx in range(n_horses)]
    horse_ability = {hid: rng.normal(0, 1) for hid in horse_ids}
    jockey_ids = [f"j{idx:03d}" for idx in range(n_jockeys)]
    jockey_skill = {jid: rng.normal(0, 1) for jid in jockey_ids}
    trainer_of = {hid: f"t{idx % 50:03d}" for idx, hid in enumerate(horse_ids)}
    sex_of = {hid: rng.choice(["牡", "牝", "セ"]) for hid in horse_ids}
    base_age = {hid: int(rng.integers(3, 7)) for hid in horse_ids}

    rows = []
    start_date = pd.Timestamp("2022-01-01")
    field = min(field_size, n_horses)

    for race_idx in range(n_races):
        race_id = f"r{race_idx:05d}"
        date = start_date + pd.Timedelta(days=race_idx * 3)
        place = rng.choice(PLACES)
        surface = rng.choice(SURFACES)
        distance = int(rng.choice(DISTANCES))
        track_condition = rng.choice(TRACK_CONDITIONS, p=TRACK_CONDITION_PROBS)

        entrants = rng.choice(horse_ids, size=field, replace=False)
        scored = []
        for umaban, hid in enumerate(entrants, start=1):
            jid = rng.choice(jockey_ids)
            ability = horse_ability[hid] + 0.4 * jockey_skill[jid]
            score = ability + rng.normal(0, 1.0)
            odds = round(max(1.1, float(rng.lognormal(mean=1.5, sigma=1.0))), 1)
            scored.append((umaban, hid, jid, score, odds))

        # popularity: rank by odds (lower odds = more popular = rank 1)
        by_odds = sorted(scored, key=lambda t: t[4])
        popularity = {hid: pop for pop, (_, hid, _, _, _) in enumerate(by_odds, start=1)}

        scored.sort(key=lambda t: -t[3])
        for rank, (umaban, hid, jid, _score, odds) in enumerate(scored, start=1):
            weight = int(rng.normal(480, 30))
            weight_diff = int(rng.normal(0, 4))
            kinryo = round(float(rng.normal(55, 2)), 1)
            rows.append(
                {
                    "race_id": race_id,
                    "date": date,
                    "place": place,
                    "surface": surface,
                    "distance": distance,
                    "track_condition": track_condition,
                    "waku": (umaban - 1) % 8 + 1,
                    "umaban": umaban,
                    "horse_id": hid,
                    "horse_name": hid,
                    "sex_age": f"{sex_of[hid]}{base_age[hid] + race_idx // 100}",
                    "kinryo": kinryo,
                    "jockey_id": jid,
                    "jockey": jid,
                    "trainer_id": trainer_of[hid],
                    "trainer": trainer_of[hid],
                    "horse_weight": f"{weight}({weight_diff:+d})",
                    "odds": odds,
                    "popularity": popularity[hid],
                    "rank": rank,
                    "last_3f": round(float(rng.normal(35, 1.5)), 1),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["date", "race_id", "rank"]).reset_index(drop=True)
