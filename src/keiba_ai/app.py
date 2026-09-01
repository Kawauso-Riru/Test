"""Streamlit UI for the horse-racing top-3-finish prediction model.

Run with:
    streamlit run src/keiba_ai/app.py
"""
from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests
import streamlit as st

from keiba_ai.features import ALL_FEATURE_COLUMNS, build_prediction_frame, build_training_frame
from keiba_ai.io import read_race_csv
from keiba_ai.model import KeibaModel, bet_type_hint, softmax_scores, train_model
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig, is_jra_race_id
from keiba_ai.synth_data import generate_synthetic_results


def fetch_with_retry(fn, *a, retries: int = 2, backoff: float = 3.0):
    """Retry transient network errors (timeouts, connection resets, 4xx/5xx)
    before giving up on one call -- mirrors scripts/predict_raceday.py's
    helper of the same name, so a single flaky request doesn't abort the
    whole "今日・明日のレースを予想" run and lose every race already
    fetched before it. Returns None (not raising) after repeated failure or
    a robots.txt denial; callers decide how to treat that."""
    for attempt in range(retries + 1):
        try:
            return fn(*a)
        except RobotsDisallowedError:
            return None
        except requests.RequestException:
            if attempt == retries:
                return None
            time.sleep(backoff * (attempt + 1))
    return None

st.set_page_config(page_title="競馬予想AI", layout="wide")
st.title("🏇 競馬予想AI")
st.caption(
    "同一レース内での相対的な強さを予測するランキングAIのデモアプリです。"
    "表示される%は同レース内での相対的な優劣(softmax)であり、実際の的中確率ではありません。"
    "娯楽・研究目的であり、賭け金の助言ではありません。"
)

REAL_MODEL_PATH = Path("models/model_dirt.joblib")
REAL_HISTORY_PATH = Path("models/history.csv")
REAL_DATA_PATH = Path("data/jra_results.csv")
REAL_OIKIRI_PATH = Path("data/oikiri.csv")
REAL_PEDIGREE_PATH = Path("data/pedigree.csv")
MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}


@st.cache_resource
def get_or_train_demo_model():
    raw = generate_synthetic_results()
    training_df = build_training_frame(raw)
    model = train_model(training_df)
    return model, training_df


@st.cache_resource
def load_real_dirt_model():
    """Load the pre-trained model/history from disk if present (the normal
    local-dev path, via `scripts/train_model.py`); otherwise train them
    on the spot from the committed data/jra_results.csv (+ data/oikiri.csv
    if present) -- e.g. on a fresh cloud deploy that only has the repo
    checked out, not a locally-trained models/ directory. Training on the
    full dataset takes well under a minute and only happens once per app
    process thanks to @st.cache_resource."""
    if REAL_MODEL_PATH.exists() and REAL_HISTORY_PATH.exists():
        model = KeibaModel.load(REAL_MODEL_PATH)
        history_df = read_race_csv(REAL_HISTORY_PATH, parse_dates=["date"])
        return model, history_df

    raw = read_race_csv(REAL_DATA_PATH)
    if REAL_OIKIRI_PATH.exists():
        oikiri = read_race_csv(REAL_OIKIRI_PATH)[["race_id", "horse_id", "training_grade"]]
        raw = raw.merge(oikiri, on=["race_id", "horse_id"], how="left")
    if REAL_PEDIGREE_PATH.exists():
        pedigree = read_race_csv(REAL_PEDIGREE_PATH)[["horse_id", "sire_id", "damsire_id"]]
        raw = raw.merge(pedigree, on="horse_id", how="left")
    training_df = build_training_frame(raw)
    fit_df = training_df[training_df["is_dirt"]]
    feature_columns = [c for c in ALL_FEATURE_COLUMNS if c not in MARKET_FEATURE_COLUMNS]
    model = train_model(fit_df, feature_columns=feature_columns)
    return model, training_df


st.sidebar.header("設定")
mode = st.sidebar.radio(
    "使い方を選択",
    [
        "デモデータで試す",
        "実データモデルを使う(学習済み)",
        "今日・明日のレースを予想",
        "CSVをアップロード",
        "URLから取得(スクレイピング)",
    ],
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
    feature_df["3着以内率(推定,%)"] = (model.predict_top3_probability(feature_df) * 100).round(1)
    return feature_df


def show_result(feature_df: pd.DataFrame) -> None:
    result = feature_df.sort_values("score", ascending=False)
    st.dataframe(
        result[["umaban", "horse_name", "jockey", "kinryo", "3着以内率(推定,%)", "相対スコア(%)"]],
        width='stretch',
        hide_index=True,
    )
    st.caption(
        "**3着以内率(推定,%)**: 過去データをもとに較正した、その馬が単独で3着以内に入る確率の推定値"
        "(馬ごとに独立。レース内で合計100%にはならない)。"
        "**相対スコア(%)**: そのレース内での相対的な強さを、フィールド内で合計100%になるよう表示したもの"
        "(的中確率そのものではない)。"
    )
    if len(result) >= 6:
        top6_probs = result["3着以内率(推定,%)"].head(6).to_numpy() / 100.0
        st.info(f"買い方の目安: {bet_type_hint(top6_probs)}")


def format_metrics(metrics: dict) -> str:
    ndcg_key = next((k for k in metrics if k.startswith("valid_ndcg@")), None)
    all3_key = next((k for k in metrics if k.startswith("valid_all_top3_in_top")), None)
    parts = [f"予測上位3頭の的中率(precision@3): {metrics['valid_precision@3']:.3f}"]
    if all3_key:
        top_k = all3_key.replace("valid_all_top3_in_top", "")
        parts.append(f"1〜3着が上位{top_k}頭に全員入る確率: {metrics[all3_key]:.3f}")
    if ndcg_key:
        parts.append(f"{ndcg_key.replace('valid_', '').upper()}: {metrics[ndcg_key]:.3f}")
    return ", ".join(parts)


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
    if not REAL_MODEL_PATH.exists() and not REAL_DATA_PATH.exists():
        st.warning(
            "学習済みモデルも学習データも見つかりません(`models/model_dirt.joblib` / "
            "`data/jra_results.csv`)。先にコマンドラインでデータを収集してください:\n\n"
            "```bash\n"
            "python scripts/scrape_jra_dirt_results.py \\\n"
            "    --start-date 20240101 --end-date 20240630 --out data/jra_results.csv\n"
            "```"
        )
        st.stop()

    if not REAL_MODEL_PATH.exists():
        st.info("学習済みモデルが見つからないため、その場でデータから学習します(数十秒かかります)。")

    model, history_df = load_real_dirt_model()
    dirt_history = history_df[history_df["surface"] == "ダート"]
    st.success(
        f"学習済み実データモデルを読み込みました "
        f"({format_metrics(model.metrics)}、"
        f"学習データ: ダート{dirt_history['race_id'].nunique()}レース、"
        f"{history_df['date'].min():%Y-%m-%d}〜{history_df['date'].max():%Y-%m-%d})"
    )

    places = sorted(dirt_history["place"].dropna().unique())
    selected_place = st.selectbox("競馬場を選択", ["すべて"] + places)
    place_filtered = dirt_history if selected_place == "すべて" else dirt_history[dirt_history["place"] == selected_place]

    # Sort by the actual date, not the race_id string: race_id encodes
    # YYYYPPKKDDRR, so string-sorting mixes different courses' place codes
    # (PP) in ahead of the day (DD) -- e.g. every 2026 小倉 (place code "10",
    # the largest of the 10 JRA codes) race would sort after *any* other
    # 2026 course's race, regardless of which one actually happened later.
    # Within a date, group by venue and then by race number (1R->12R) --
    # otherwise same-day races interleave in whatever order they happened to
    # land in the CSV, which reads as scattered rather than organized.
    race_order = place_filtered[["race_id", "date", "place"]].drop_duplicates("race_id").copy()
    race_order["race_no"] = race_order["race_id"].astype(str).str[-2:].astype(int)
    race_order = race_order.sort_values(["date", "place", "race_no"])
    race_ids = race_order["race_id"].tolist()[-50:]

    if not race_ids:
        st.warning("選択した競馬場のダートレースが見つかりませんでした。")
        st.stop()

    race_labels = race_label_lookup(place_filtered, race_ids)
    chosen_race = st.selectbox(
        f"ダートレースを選択(直近{len(race_ids)}件)", race_ids,
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
    # sire_id/damsire_id are static per horse, so (unlike horse_*_before
    # stats) they only reach this replay via the row itself -- present only
    # if data/pedigree.csv existed when history_df was built.
    shutuba_cols += [c for c in ("sire_id", "damsire_id") if c in race_rows.columns]
    shutuba = race_rows[shutuba_cols].copy()
    # Only use history strictly before this race's date -- otherwise the
    # race's own result would leak into the stats used to predict it.
    history_for_pred = history_df[history_df["date"] < info["date"]]

    feature_df = add_predictions(model, build_prediction_frame(shutuba, history_for_pred))
    show_result(feature_df)

    with st.expander("実際の着順と比較"):
        predicted_rank = feature_df[["umaban", "score"]].copy()
        predicted_rank["predicted_rank"] = predicted_rank["score"].rank(ascending=False, method="min").astype(int)
        actual = race_rows[["umaban", "horse_name", "rank"]].merge(
            predicted_rank[["umaban", "predicted_rank"]], on="umaban", how="left",
        )
        # "rank" is text (a DNF/scratch shows up as e.g. "中止" rather than a
        # number), so sorting the column directly sorts lexicographically
        # ("1","10","11",...,"2","3",...) instead of by finishing position.
        # Sort by a numeric-coerced key instead; non-finishers (NaN) sort last.
        actual = actual.assign(_rank_numeric=pd.to_numeric(actual["rank"], errors="coerce"))
        actual = actual.sort_values("_rank_numeric").drop(columns="_rank_numeric")
        actual = actual.rename(columns={"rank": "着順", "predicted_rank": "予測順位"})
        st.dataframe(
            actual[["着順", "umaban", "horse_name", "予測順位"]],
            width='stretch', hide_index=True,
        )

elif mode == "今日・明日のレースを予想":
    st.write(
        "学習済みモデルで、これから開催されるレースを予想します。"
        "race.netkeiba.com の開催カードを毎回取得するため、レース結果ではなく"
        "**まだ結果の出ていない、これから走るレース**が対象です。"
        "開催の数日前〜当日にカードが発表されてから使えます(発表前の日付は空振りになります)。"
    )
    if not REAL_MODEL_PATH.exists() and not REAL_DATA_PATH.exists():
        st.warning(
            "学習済みモデルも学習データも見つかりません(`models/model_dirt.joblib` / "
            "`data/jra_results.csv`)。先にコマンドラインでデータを収集してください:\n\n"
            "```bash\n"
            "python scripts/scrape_jra_dirt_results.py \\\n"
            "    --start-date 20240101 --end-date 20240630 --out data/jra_results.csv\n"
            "```"
        )
        st.stop()

    if not REAL_MODEL_PATH.exists():
        st.info("学習済みモデルが見つからないため、その場でデータから学習します(数十秒かかります)。")

    model, history_df = load_real_dirt_model()
    st.success(f"学習済み実データモデルを読み込みました ({format_metrics(model.metrics)})")
    pedigree_df = (
        read_race_csv(REAL_PEDIGREE_PATH)[["horse_id", "sire_id", "damsire_id"]]
        if REAL_PEDIGREE_PATH.exists() else None
    )

    today = datetime.date.today()
    if "predict_date" not in st.session_state:
        st.session_state["predict_date"] = today

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("今日"):
            st.session_state["predict_date"] = today
    with col2:
        if st.button("明日"):
            st.session_state["predict_date"] = today + datetime.timedelta(days=1)
    with col3:
        target_date = st.date_input("予想したい開催日", key="predict_date")

    dirt_only = st.checkbox("ダートレースのみ予想する(このモデルはダート特化です)", value=True)
    contact = st.text_input(
        "連絡先メールアドレス",
        help="スクレイパーのUser-Agentに埋め込まれ、アクセス元を示すために使われます(必須)。",
    )
    min_interval = st.slider("最小リクエスト間隔(秒)", 1.0, 5.0, 1.5)

    if st.button("この日のレースを予想する", type="primary"):
        if not contact:
            st.error("連絡先メールアドレスを入力してください。")
            st.stop()

        scraper = PoliteScraper(
            ScraperConfig(
                user_agent=f"keiba-ai-research-bot/0.1 (+contact: {contact})",
                min_interval_sec=min_interval,
            )
        )
        date_str = target_date.strftime("%Y%m%d")
        with st.spinner("開催レース一覧を取得中..."):
            races = fetch_with_retry(scraper.list_upcoming_races_for_date, date_str)

        if races is None:
            st.error("開催レース一覧の取得に失敗しました(通信エラー、またはrobots.txtによる禁止)。しばらくしてからもう一度お試しください。")
            st.stop()

        jra_races = [r for r in races if is_jra_race_id(r["race_id"])]
        if not jra_races:
            st.warning(
                f"{target_date:%Y年%m月%d日}の中央競馬レースが見つかりませんでした。"
                "開催がない日か、まだ出馬表が発表されていない可能性があります"
                "(通常、開催の数日前から発表されます)。"
            )
            st.stop()

        progress = st.progress(0.0)
        status = st.empty()
        race_results = []
        for i, race in enumerate(jra_races):
            race_id, place = race["race_id"], race["place"]
            status.text(f"取得中... {place} {int(race_id[-2:])}R ({i + 1}/{len(jra_races)})")
            parsed = fetch_with_retry(scraper.fetch_shutuba, scraper.shutuba_url(race_id))
            if parsed is None:
                progress.progress((i + 1) / len(jra_races))
                continue

            if parsed["entries"]:
                meta = parsed["meta"]
                surface = meta.get("surface", "")
                if not (dirt_only and surface != "ダート"):
                    shutuba_df = pd.DataFrame(parsed["entries"])
                    shutuba_df["surface"] = surface
                    shutuba_df["distance"] = meta.get("distance")
                    shutuba_df["track_condition"] = meta.get("track_condition", "")
                    shutuba_df["place"] = meta.get("place") or place
                    shutuba_df["race_name"] = meta.get("race_name", "")

                    oikiri_entries = fetch_with_retry(scraper.fetch_oikiri, scraper.oikiri_url(race_id)) or []
                    if oikiri_entries:
                        oikiri_df = pd.DataFrame(oikiri_entries)[["horse_id", "training_grade"]]
                        shutuba_df = shutuba_df.merge(oikiri_df, on="horse_id", how="left")

                    if pedigree_df is not None:
                        shutuba_df = shutuba_df.merge(pedigree_df, on="horse_id", how="left")

                    feature_df = add_predictions(model, build_prediction_frame(shutuba_df, history_df))
                    race_no = int(race_id[-2:])
                    label = f"{place} {race_no}R  {meta.get('race_name', '')} ({surface}{meta.get('distance', '?')}m)"
                    race_results.append((race_no, label, feature_df))
            progress.progress((i + 1) / len(jra_races))

        status.empty()
        progress.empty()

        if not race_results:
            st.warning(
                "予想できるレースがありませんでした。出馬表がまだ確定していないか、"
                "ダート指定で該当レースがなかった可能性があります。"
            )
            st.stop()

        race_results.sort(key=lambda r: r[0])
        st.success(f"{target_date:%Y年%m月%d日} のレースを{len(race_results)}件予想しました。")
        for _, label, feature_df in race_results:
            with st.expander(label):
                show_result(feature_df)

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
        shutuba_df["race_name"] = smeta.get("race_name", "")

        feature_df = add_predictions(model, build_prediction_frame(shutuba_df, training_df))
        show_result(feature_df)
