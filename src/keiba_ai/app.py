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
from keiba_ai.model import train_model
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig
from keiba_ai.synth_data import generate_synthetic_results

st.set_page_config(page_title="競馬予想AI", layout="wide")
st.title("🏇 競馬予想AI")
st.caption("複勝圏内(3着以内)に入る確率を予測するデモアプリです。娯楽・研究目的であり、賭け金の助言ではありません。")


@st.cache_resource
def get_or_train_demo_model():
    raw = generate_synthetic_results()
    training_df = build_training_frame(raw)
    model = train_model(training_df)
    return model, training_df


st.sidebar.header("設定")
mode = st.sidebar.radio(
    "使い方を選択",
    ["デモデータで試す", "CSVをアップロード", "URLから取得(スクレイピング)"],
)
st.sidebar.markdown(
    "---\n"
    "**注意:** スクレイピング機能を使う場合は、対象サイトの利用規約・robots.txt を必ず確認し、"
    "アクセス頻度を抑えてご利用ください。"
)


def show_result(feature_df: pd.DataFrame) -> None:
    result = feature_df.sort_values("top3_probability(%)", ascending=False)
    st.dataframe(
        result[["umaban", "horse_name", "jockey", "kinryo", "top3_probability(%)"]],
        use_container_width=True,
        hide_index=True,
    )


if mode == "デモデータで試す":
    with st.spinner("デモモデルを学習中..."):
        model, training_df = get_or_train_demo_model()
    st.success(f"デモモデル学習完了 (検証AUC: {model.metrics['valid_auc']:.3f})")

    race_ids = sorted(training_df["race_id"].unique())[-20:]
    chosen_race = st.selectbox("レースを選択(合成データ・直近20件)", race_ids)
    shutuba_cols = [
        "horse_id", "jockey_id", "umaban", "waku", "horse_name", "jockey",
        "sex_age", "kinryo", "horse_weight", "surface", "distance", "track_condition", "place",
    ]
    shutuba = training_df[training_df["race_id"] == chosen_race][shutuba_cols].copy()
    history = training_df[training_df["race_id"] != chosen_race]

    feature_df = build_prediction_frame(shutuba, history)
    feature_df["top3_probability(%)"] = (model.predict(feature_df) * 100).round(1)
    show_result(feature_df)
    st.caption("※ 合成生成した架空のレースです。実データではありません。")

elif mode == "CSVをアップロード":
    st.write("学習用の過去成績CSVと、予測対象の出馬表CSVをそれぞれアップロードしてください。列名は keiba_ai.parser の出力に合わせてください。")
    hist_file = st.file_uploader("過去成績CSV", type="csv", key="hist")
    shutuba_file = st.file_uploader("出馬表CSV", type="csv", key="shutuba")

    if hist_file and shutuba_file:
        raw = pd.read_csv(hist_file)
        training_df = build_training_frame(raw)
        with st.spinner("モデルを学習中..."):
            model = train_model(training_df)
        st.success(f"学習完了 (検証AUC: {model.metrics['valid_auc']:.3f})")

        shutuba = pd.read_csv(shutuba_file)
        feature_df = build_prediction_frame(shutuba, training_df)
        feature_df["top3_probability(%)"] = (model.predict(feature_df) * 100).round(1)
        show_result(feature_df)

else:
    st.write(
        "過去レース結果ページの URL(複数可)と、予測したい出馬表ページの URL を入力してください。"
        "サイトの利用規約に従い、自己責任でご利用ください。少数のURLでは学習データが少なすぎて"
        "精度は出ません(デモ用途を想定)。"
    )
    result_urls_text = st.text_area("過去レース結果ページ URL(1行1URL)")
    shutuba_url = st.text_input("予測したい出馬表ページ URL")
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
                    rows.append(entry)
                progress.progress((i + 1) / len(result_urls))
        except RobotsDisallowedError as exc:
            st.error(f"robots.txt によりアクセスが禁止されています: {exc}")
            st.stop()

        if not rows:
            st.error("結果ページから有効な行を取得できませんでした。ページ構造を確認してください。")
            st.stop()

        raw = pd.DataFrame(rows)
        raw["date"] = pd.Timestamp.today().normalize()
        training_df = build_training_frame(raw)
        with st.spinner("モデルを学習中..."):
            model = train_model(training_df)
        st.success(f"学習完了 (検証AUC: {model.metrics['valid_auc']:.3f})")

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
        shutuba_df["place"] = smeta.get("race_name", "")

        feature_df = build_prediction_frame(shutuba_df, training_df)
        feature_df["top3_probability(%)"] = (model.predict(feature_df) * 100).round(1)
        show_result(feature_df)
