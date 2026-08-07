"""Streamlit UI for the horse-racing top-3-finish prediction model.

Run with:
    streamlit run src/keiba_ai/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from keiba_ai.features import build_prediction_frame, build_training_frame
from keiba_ai.io import read_race_csv
from keiba_ai.model import KeibaModel, softmax_scores, train_model
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig
from keiba_ai.synth_data import generate_synthetic_results

st.set_page_config(page_title="競馬予想AI", layout="wide")
st.title("🏇 競馬予想AI")
st.caption(
    "同一レース内での相対的な強さを予測するランキングAIのデモアプリです。"
    "表示される%は同レース内での相対的な優劣(softmax)であり、実際の的中確率ではありません。"
    "娯楽・研究目的であり、賭け金の助言ではありません。"
)

REAL_MODEL_PATH = Path("models/model_dirt.joblib")
REAL_HISTORY_PATH = Path("models/history.csv")


@st.cache_resource
def get_or_train_demo_model():
    raw = generate_synthetic_results()
    training_df = build_training_frame(raw)
    model = train_model(training_df)
    return model, training_df


@st.cache_resource
def load_real_dirt_model():
    model = KeibaModel.load(REAL_MODEL_PATH)
    history_df = read_race_csv(REAL_HISTORY_PATH, parse_dates=["date"])
    return model, history_df


st.sidebar.header("設定")
mode = st.sidebar.radio(
    "使い方を選択",
    ["デモデータで試す", "実データモデルを使う(学習済み)", "CSVをアップロード", "URLから取得(スクレイピング)"],
)
st.sidebar.markdown(
    "---\n"
    "**注意:** スクレイピング機能を使う場合は、対象サイトの利用規約・robots.txt を必ず確認し、"
    "アクセス頻度を抑えてご利用ください。"
)


def add_predictions(model, feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.copy()
    scores = model.predict(feature_df)
    feature_df["score"] = scores
    feature_df["相対スコア(%)"] = (softmax_scores(scores) * 100).round(1)
    return feature_df


def show_result(feature_df: pd.DataFrame) -> None:
    result = feature_df.sort_values("score", ascending=False)
    st.dataframe(
        result[["umaban", "horse_name", "jockey", "kinryo", "相対スコア(%)"]],
        width='stretch',
        hide_index=True,
    )


def format_metrics(metrics: dict) -> str:
    return f"NDCG@3: {metrics['valid_ndcg@3']:.3f}, 予測上位3頭の的中率(precision@3): {metrics['valid_precision@3']:.3f}"


def race_label_lookup(df: pd.DataFrame, race_ids) -> dict:
    """race_id -> "2026/8/7 中山 11R" style label, for a readable selectbox.

    Race number is read off the last 2 digits of race_id (netkeiba's
    YYYYPPKKDDRR convention) since it isn't stored as its own column.
    """
    preview = df[df["race_id"].isin(race_ids)].drop_duplicates("race_id").set_index("race_id")
    labels = {}
    for race_id in race_ids:
        info = preview.loc[race_id]
        race_no = int(str(race_id)[-2:])
        labels[race_id] = f"{info['date'].year}/{info['date'].month}/{info['date'].day} {info['place']} {race_no}R"
    return labels


if mode == "デモデータで試す":
    with st.spinner("デモモデルを学習中..."):
        model, training_df = get_or_train_demo_model()
    st.success(f"デモモデル学習完了 ({format_metrics(model.metrics)})")

    race_ids = sorted(training_df["race_id"].unique())[-20:]
    race_labels = race_label_lookup(training_df, race_ids)
    chosen_race = st.selectbox(
        "レースを選択(合成データ・直近20件)", race_ids, format_func=lambda rid: race_labels[rid]
    )
    shutuba_cols = [
        "horse_id", "jockey_id", "trainer_id", "umaban", "waku", "horse_name", "jockey",
        "sex_age", "kinryo", "horse_weight", "surface", "distance", "track_condition", "place",
        "popularity", "odds",
    ]
    shutuba = training_df[training_df["race_id"] == chosen_race][shutuba_cols].copy()
    history = training_df[training_df["race_id"] != chosen_race]

    feature_df = add_predictions(model, build_prediction_frame(shutuba, history))
    show_result(feature_df)
    st.caption("※ 合成生成した架空のレースです。実データではありません。")

elif mode == "実データモデルを使う(学習済み)":
    if not REAL_MODEL_PATH.exists() or not REAL_HISTORY_PATH.exists():
        st.warning(
            "学習済みモデルが見つかりません(`models/model_dirt.joblib` / "
            "`models/history.csv`)。先にコマンドラインで収集・学習してください:\n\n"
            "```bash\n"
            "python scripts/scrape_jra_dirt_results.py \\\n"
            "    --start-date 20240101 --end-date 20240630 --out data/jra_results.csv\n"
            "python scripts/train_model.py --data data/jra_results.csv --dirt-only \\\n"
            "    --model-out models/model_dirt.joblib --history-out models/history.csv\n"
            "```"
        )
        st.stop()

    model, history_df = load_real_dirt_model()
    dirt_history = history_df[history_df["surface"] == "ダート"]
    st.success(
        f"学習済み実データモデルを読み込みました "
        f"({format_metrics(model.metrics)}、"
        f"学習データ: ダート{dirt_history['race_id'].nunique()}レース、"
        f"{history_df['date'].min():%Y-%m-%d}〜{history_df['date'].max():%Y-%m-%d})"
    )

    race_ids = sorted(dirt_history["race_id"].unique())[-50:]
    race_labels = race_label_lookup(dirt_history, race_ids)
    chosen_race = st.selectbox(
        "ダートレースを選択(履歴データの直近50件)", race_ids,
        index=len(race_ids) - 1, format_func=lambda rid: race_labels[rid],
    )
    race_rows = dirt_history[dirt_history["race_id"] == chosen_race]
    info = race_rows.iloc[0]
    st.caption(f"{info['date']:%Y-%m-%d} {info['place']} {info['surface']}{info['distance']:.0f}m {info['track_condition']}")

    shutuba_cols = [
        "horse_id", "jockey_id", "trainer_id", "umaban", "waku", "horse_name", "jockey",
        "sex_age", "kinryo", "horse_weight", "surface", "distance", "track_condition", "place",
        "popularity", "odds",
    ]
    shutuba = race_rows[shutuba_cols].copy()
    # Only use history strictly before this race's date -- otherwise the
    # race's own result would leak into the stats used to predict it.
    history_for_pred = history_df[history_df["date"] < info["date"]]

    feature_df = add_predictions(model, build_prediction_frame(shutuba, history_for_pred))
    show_result(feature_df)

    with st.expander("実際の着順と比較"):
        actual = race_rows[["umaban", "horse_name", "rank"]].sort_values("rank")
        st.dataframe(actual, width='stretch', hide_index=True)

elif mode == "CSVをアップロード":
    st.write("学習用の過去成績CSVと、予測対象の出馬表CSVをそれぞれアップロードしてください。列名は keiba_ai.parser の出力に合わせてください。")
    hist_file = st.file_uploader("過去成績CSV", type="csv", key="hist")
    shutuba_file = st.file_uploader("出馬表CSV", type="csv", key="shutuba")

    if hist_file and shutuba_file:
        raw = read_race_csv(hist_file)
        training_df = build_training_frame(raw)
        with st.spinner("モデルを学習中..."):
            model = train_model(training_df)
        st.success(f"学習完了 ({format_metrics(model.metrics)})")

        shutuba = read_race_csv(shutuba_file)
        feature_df = add_predictions(model, build_prediction_frame(shutuba, training_df))
        show_result(feature_df)

else:
    st.write(
        "過去レース結果ページの URL(複数可)と、予測したい出馬表ページの URL を入力してください。"
        "サイトの利用規約に従い、自己責任でご利用ください。少数のURLでは学習データが少なすぎて"
        "精度は出ません(デモ用途を想定)。"
    )
    result_urls_text = st.text_area("過去レース結果ページ URL(1行1URL)")
    shutuba_url = st.text_input("予測したい出馬表ページ URL")
    shutuba_place = st.text_input(
        "出馬表の開催場(例: 中山)",
        value="",
        help="出馬表ページ(race.netkeiba.com)からは開催場を自動取得できないため、手入力で補ってください。",
    )
    min_interval = st.slider("最小リクエスト間隔(秒)", 1.0, 10.0, 3.0)

    if st.button("取得して予測"):
        result_urls = [u.strip() for u in result_urls_text.splitlines() if u.strip()]
        if not shutuba_url or not result_urls:
            st.error("結果ページURLと出馬表URLの両方を入力してください。")
            st.stop()

        scraper = PoliteScraper(ScraperConfig(min_interval_sec=min_interval))
        rows = []
        progress = st.progress(0.0)
        try:
            for i, url in enumerate(result_urls):
                parsed = scraper.fetch_race_result(url)
                meta = parsed["meta"]
                for entry in parsed["entries"]:
                    entry.update(meta)
                    entry["race_id"] = url
                    # db.netkeiba.com result pages carry their own calendar date
                    # in meta; fall back to today only if that couldn't be parsed.
                    entry.setdefault("date", pd.Timestamp.today().strftime("%Y-%m-%d"))
                    rows.append(entry)
                progress.progress((i + 1) / len(result_urls))
        except RobotsDisallowedError as exc:
            st.error(f"robots.txt によりアクセスが禁止されています: {exc}")
            st.stop()

        if not rows:
            st.error("結果ページから有効な行を取得できませんでした。ページ構造を確認してください。")
            st.stop()

        raw = pd.DataFrame(rows)
        training_df = build_training_frame(raw)
        with st.spinner("モデルを学習中..."):
            model = train_model(training_df)
        st.success(f"学習完了 ({format_metrics(model.metrics)})")

        try:
            shutuba_parsed = scraper.fetch_shutuba(shutuba_url)
        except RobotsDisallowedError as exc:
            st.error(f"robots.txt によりアクセスが禁止されています: {exc}")
            st.stop()

        shutuba_df = pd.DataFrame(shutuba_parsed["entries"])
        smeta = shutuba_parsed["meta"]
        shutuba_df["surface"] = smeta.get("surface", "")
        shutuba_df["distance"] = smeta.get("distance")
        shutuba_df["track_condition"] = smeta.get("track_condition", "")
        shutuba_df["place"] = smeta.get("place") or shutuba_place

        feature_df = add_predictions(model, build_prediction_frame(shutuba_df, training_df))
        show_result(feature_df)
