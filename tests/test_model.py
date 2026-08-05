from keiba_ai.features import build_prediction_frame, build_training_frame
from keiba_ai.model import train_model
from keiba_ai.synth_data import generate_synthetic_results


def test_train_model_beats_random_guessing():
    raw = generate_synthetic_results(n_horses=80, n_jockeys=15, n_races=150, field_size=10, seed=1)
    training_df = build_training_frame(raw)
    model = train_model(training_df, num_boost_round=100)

    # Synthetic data has real (ability + jockey skill) signal, so a trained
    # model should clearly beat AUC 0.5 (random guessing).
    assert model.metrics["valid_auc"] > 0.55


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

    assert len(preds) == len(shutuba)
    assert ((preds >= 0) & (preds <= 1)).all()
