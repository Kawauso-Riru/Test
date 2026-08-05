import numpy as np

from keiba_ai.features import build_prediction_frame, build_training_frame
from keiba_ai.model import softmax_scores, train_model
from keiba_ai.synth_data import generate_synthetic_results


def test_train_model_beats_random_guessing():
    raw = generate_synthetic_results(n_horses=80, n_jockeys=15, n_races=150, field_size=10, seed=1)
    training_df = build_training_frame(raw)
    model = train_model(training_df, num_boost_round=100)

    # Synthetic data has real (ability + jockey skill) signal. A random
    # top-3 guess out of a 10-horse field would score precision@3 ~= 0.3;
    # a trained ranker should clearly beat that.
    assert model.metrics["valid_precision@3"] > 0.4
    assert model.metrics["valid_ndcg@3"] > 0.5


def test_predict_on_upcoming_race_shape():
    raw = generate_synthetic_results(n_horses=60, n_jockeys=10, n_races=120, field_size=8, seed=2)
    training_df = build_training_frame(raw)
    model = train_model(training_df, num_boost_round=50)

    last_race_id = training_df.sort_values("date")["race_id"].iloc[-1]
    shutuba_cols = [
        "horse_id", "jockey_id", "umaban", "waku", "horse_name", "jockey",
        "sex_age", "kinryo", "horse_weight", "surface", "distance", "track_condition", "place",
    ]
    shutuba = training_df[training_df["race_id"] == last_race_id][shutuba_cols]
    history = training_df[training_df["race_id"] != last_race_id]

    feature_df = build_prediction_frame(shutuba, history)
    preds = model.predict(feature_df)

    # Ranking scores are unbounded (not a [0,1] probability) -- only their
    # relative order within a race is meaningful.
    assert len(preds) == len(shutuba)
    assert np.isfinite(preds).all()

    probs = softmax_scores(preds)
    assert np.isclose(probs.sum(), 1.0)
    assert ((probs >= 0) & (probs <= 1)).all()
