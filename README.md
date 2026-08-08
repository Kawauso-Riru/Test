# 競馬予想AI (keiba-ai)

過去のレース結果から、同一レース内での各馬の相対的な強さを予測する
学習用の最小構成AIパイプラインです。データ収集(スクレイピング)・
特徴量エンジニアリング・LightGBM(LambdaMARTランキングモデル)・
Streamlit UI までを一通り含みます。

**⚠️ 重要な注意事項**
- 実サイトへのスクレイピング機能を含みますが、**対象サイトの利用規約(ToS)・
  robots.txt を必ず確認し、遵守できる範囲でのみ使用してください。** 本リポジトリの
  スクレイパーは既定でリクエスト間隔を空け、robots.txt が禁止しているURLへの
  アクセスは拒否しますが、これは免責にはなりません。利用は自己責任でお願いします。
- 予測結果は娯楽・研究目的のものであり、**賭け金や馬券購入の助言ではありません。**
- 同梱の合成データ (`synth_data.py`) は実在のレース・馬・騎手とは無関係の
  架空データです。ネットワークなしでパイプライン全体を試すために用意しています。
- `data/jra_results.csv`(収集済みの実データ)はリポジトリに含まれています。
  `models/` 以下(学習済みモデル・履歴)は `data/jra_results.csv` から数秒で
  再現できるため含めていません -- 詳しくは「自動化」の節を参照してください。

## 構成

```
src/keiba_ai/
  parser.py      # HTMLパース(純粋関数、ネットワーク非依存)
  scraper.py      # robots.txt尊重・レート制限付きのネットワーク層
  io.py           # CSV読み込み共通ヘルパー(ID列のゼロ落ち対策、下記参照)
  synth_data.py   # オフラインデモ/テスト用の合成レースデータ生成
  features.py     # 未来情報リークなしの特徴量エンジニアリング
  model.py        # LightGBMモデル(LambdaMARTランキング)の学習・保存・推論
  app.py          # Streamlit UI
scripts/
  generate_demo_data.py       # 合成データをCSVに書き出す
  scrape_jra_dirt_results.py  # 中央競馬(JRA10場)の実レース結果を収集
  update_dataset.py            # 既存データセットを差分更新して再学習(自動化用)
  predict_raceday.py           # 指定日の中央競馬レースをまとめて予測
  tune_hyperparams.py          # LightGBMハイパーパラメータのランダムサーチ
  train_model.py               # CSVからモデルを学習(--dirt-onlyでダート特化)
  predict_race.py              # 出馬表(CSV/URL)から1レース分を予測
tests/                          # pytest (parser/features/model/io/scraperのユニットテスト)
.github/workflows/
  weekly_update.yml             # 週次でupdate_dataset.pyを実行するGitHub Actions
```

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## クイックスタート(合成データで一通り試す)

```bash
# 1. 合成レースデータを生成(実サイトへのアクセス不要)
python scripts/generate_demo_data.py --out data/demo_results.csv

# 2. モデルを学習
python scripts/train_model.py --data data/demo_results.csv \
    --model-out models/model.joblib --history-out models/history.csv

# 3. 出馬表CSV(過去成績と同じ列構成)を用意して予測
python scripts/predict_race.py --shutuba-csv data/sample_shutuba.csv \
    --model models/model.joblib --history models/history.csv
```

または Streamlit UI で:

```bash
streamlit run src/keiba_ai/app.py
```

サイドバーで「デモデータで試す」を選べば、その場でモデルを学習して
架空レースの予測結果を確認できます。`models/model_dirt.joblib` /
`models/history.csv` を用意済みなら「実データモデルを使う(学習済み)」
モードで、再学習なしに実データの予測をすぐ試せます(下記参照)。

## 実データを使う場合(中央競馬・ダート特化)

```bash
# 1. 中央競馬(JRA10場)のレース結果を収集(芝・ダート両方。理由は下記)
#    月ごとに分割して実行し、リクエスト間隔を空けるのを推奨(下の「収集のコツ」参照)
python scripts/scrape_jra_dirt_results.py \
    --start-date 20240101 --end-date 20240630 \
    --out data/jra_results.csv --max-races 2000 \
    --contact your-email@example.com

# 2. ダート限定でモデルを学習(馬/騎手の履歴特徴量には芝レースも活用しつつ、
#    学習対象の行だけダートレースに絞る)。model_dirt.joblib という名前で
#    保存すると、Streamlit の「実データモデルを使う」モードがそのまま拾う。
python scripts/train_model.py --data data/jra_results.csv --dirt-only \
    --model-out models/model_dirt.joblib --history-out models/history.csv

# 3. 予測(CLI、または streamlit run src/keiba_ai/app.py の
#    「実データモデルを使う(学習済み)」モードでも同じモデルを使える)
python scripts/predict_race.py --shutuba-csv data/sample_shutuba.csv \
    --model models/model_dirt.joblib --history models/history.csv
```

`scrape_jra_dirt_results.py` は `db.netkeiba.com` の日別レース一覧ページを
起点に、レースIDの2桁の場コード(01〜10)で中央競馬10場(札幌・函館・福島・
新潟・東京・中山・中京・京都・阪神・小倉)のみを抽出し、地方競馬(NAR)を
除外します。**芝レースもあえて収集・保存します**(馬/騎手の「全体的な実力」の
参考になり、出走間隔などの特徴量にも必要なため)。ダートへの特化は
`train_model.py --dirt-only` で学習対象の行を絞ることで行います。

### 収集のコツ(長期間のデータを集める場合)

一度に長い期間を `--start-date`/`--end-date` で指定すると、途中でネット
ワークエラーが起きた際に再実行がゼロからやり直しになりやすい(20レース
ごとにCSVへチェックポイント保存はされますが)ので、**月単位など小さい
期間に分けて複数回実行し、最後に結合する**方式を推奨します。同じ
`--cache-dir`(既定: `data/cache/netkeiba`)を使い回せば、取得済みページは
再フェッチされません。

```bash
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  python scripts/scrape_jra_dirt_results.py \
      --start-date 2024${m}01 --end-date 2024${m}31 \
      --out data/chunks/jra_2024${m}.csv --max-races 400 \
      --contact your-email@example.com
done
python -c "
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in glob.glob('data/chunks/jra_2024*.csv')])
df.drop_duplicates(subset=['race_id','umaban']).to_csv('data/jra_results.csv', index=False)
"
```

学習データが少ない(数週間・数百レース程度)と、`horse_dirt_*` (馬のダート
限定成績)特徴量が十分に貯まらず精度が伸び悩みます。2年半強(ダート4,199
レース、2024年1月〜2026年8月)で学習した現在のランキングモデルでは、
**held-out precision@3が0.468**(モデルの予測上位3頭のうち約47%が実際に
3着以内)で、ランダムに3頭を選んだ場合の期待値(約0.2、フィールドサイズ
約15頭として3/15)を2倍以上上回っています。

⚠️ 実サイトへのアクセスを行うため、事前に対象サイトの利用規約を確認してください。

## モデルについて

- **目的関数**: LightGBMの `lambdarank`(LambdaMART)によるランキング学習。
  レースは「同じレース内の馬同士の相対順位」を当てる問題なので、各馬を
  独立に0/1分類するより、レース単位でグループ化してランキングを直接
  最適化する方が本来のタスクに合っています。目的変数は着順に基づく
  段階的な relevance ラベル(1着=3, 2着=2, 3着=1, それ以外=0)です。
  `model.predict()` の出力は**そのレース内でのみ意味を持つ相対スコア**
  (0〜1の確率ではありません)。`softmax_scores()` でレース内softmaxを
  取ると、フィールド内で合計100%になる「相対スコア(%)」として表示できます
  (的中確率そのものではない点に注意)。
- **評価指標**: `GroupShuffleSplit` でレース単位に分割し(同一レースの馬が
  学習/検証に分かれて跨がらないように)、held-outレースで
  **NDCG@3**(ランキング品質)と**precision@3**(予測上位3頭のうち実際に
  3着以内だった割合)を表示します。
- 基本特徴量: 斤量・枠番・馬番・馬体重(増減)・年齢・距離・馬場状態・コース・
  性別に加え、**その馬/騎手のその時点までの成績を集計した「リークなし」特徴量**
  (出走数・勝率・複勝率・平均着順・前走からの間隔)を使用します。学習時は
  `groupby + cumsum` で各レースより前のデータのみから計算し、未来の結果が
  紛れ込まないようにしています。予測時(`build_prediction_frame`)には、その
  馬/騎手の**直近レースの結果まで織り込んだ**最新の統計を使います。
- **ダート特化特徴量**: 上記に加え、`horse_dirt_*` / `jockey_dirt_*`
  (ダートレースのみに絞った出走数・勝率・複勝率・平均着順)を持ちます。
  芝レースを挟んでも正しくスキップされ、その馬/騎手の直近の**ダートでの**
  実績のみを反映します(`tests/test_features.py` で検証)。芝適性とダート
  適性は必ずしも一致しないため、両者を区別して学習させています。
- **距離帯・コースバイアス特徴量**: `distance_band`(短距離≦1400m/マイル
  ≦1800m/長距離)のカテゴリと、`course_waku_bias_*`
  ((開催場, 芝/ダート, 距離帯, 枠番)の組み合わせごとの、リークなし展開勝率・
  複勝率・平均着順)を持ちます。これは特定の馬や騎手の実力とは独立に、
  「このコース・距離ではこの枠が構造的に有利/不利」といったコース固有の
  傾向を捉えるための特徴量です。
- **脚質(先行度合い)特徴量**: `horse_early_position_ratio_before` /
  `horse_dirt_early_position_ratio_before`。結果ページの「通過」列
  (コーナー通過順位)からその馬の歴代の先行度合い(0=常に先頭付近、
  1=常に後方)をリークなしで集計したものです。コースごとの脚質有利/不利は
  別途集計テーブルを作るのではなく、この連続値とコース系カテゴリ特徴量の
  交互作用としてLightGBMの木構造に学習させる設計です。
- **調教師(trainer)特徴量**: `trainer_runs_before` / `trainer_win_rate_before`
  / `trainer_top3_rate_before`(全体・ダート限定の両方)。jockey_*と全く同じ
  パターンで、調教師IDごとにリークなし展開集計しています。
- **馬×条件別のダート実績**: `horse_dirt_track_condition_*` /
  `horse_dirt_place_*` / `horse_dirt_distance_*`。`course_waku_bias_*`が
  「その条件でのフィールド全体の傾向」であるのに対し、こちらは**その馬自身
  が今回と同じ馬場状態・開催場所・距離のダート戦でどう走ってきたか**を
  リークなし展開集計したものです(いずれもダート戦に限定。`horse_dirt_*`
  同様、芝適性がそのまま転用できるとは限らないため)。学習時と同じ手法で
  `horse_dirt_distance_avg_rank_before`が特徴量重要度で上位10位以内に入る
  など、実際にモデルの予測に活用されていることを確認済みです。
  `track_condition`は出馬表発表直後は未確定(空欄)のことが多く、その場合は
  `popularity_numeric`/`odds_numeric`と同様に欠損(NaN)として扱われます。
- **人気・オッズ特徴量(既定では無効)**: `popularity_numeric` / `odds_numeric`
  として実装済みですが、`train_model.py`は**既定でこれらを除外**します。
  理由は実験で判明した実務上の落とし穴です: 最終オッズ・人気は市場(＝他の
  馬券購入者全員の判断)を丸ごと特徴量に取り込むことになるため単体では
  圧倒的に強い予測力を持ちますが(下表参照)、レース数日前の出馬表では
  まだ確定しておらず("**"のようなプレースホルダ)、`predict_raceday.py`が
  想定する「直前ではないタイミングでの予測」ではほぼ全レースで欠損します。
  欠損すると学習時にこの特徴量に頼り切ったモデルの予測がほぼ一様になって
  しまうため、既定では除外し、`--include-market-features`を明示的に渡した
  ときだけ使う設計にしています。

  | 特徴量セット | held-out precision@3 | 備考 |
  |---|---|---|
  | 人気・オッズ**あり** | 0.553 | 直前予測専用。オッズ確定前は使えない |
  | 人気・オッズ**なし(既定)** | 0.468 | いつでも使える。実運用のデフォルト |

- **ハイパーパラメータ**: `scripts/tune_hyperparams.py`
  (`learning_rate`/`num_leaves`/`min_data_in_leaf`/`feature_fraction`/
  `bagging_fraction`/`bagging_freq`/`lambda_l1`/`lambda_l2`の乱数サーチ)で
  見つけた値を`keiba_ai.model.DEFAULT_PARAMS`の既定値にしています。
- 1年分の実データで学習したところ、上記のダート特化・コースバイアス・
  脚質・調教師の各特徴量群はいずれも特徴量重要度の上位〜中位に入り、実際に
  予測に寄与していることを確認しました。
- **⚠️ 修正した重大なバグ**: 騎手・調教師IDはゼロ埋め("01209"など)された
  文字列ですが、`pd.read_csv`はこれを全桁数字の列とみなして`int64`型に
  自動推論し、先頭のゼロを黙って落とします(`"01209"` → `1209`)。保存前の
  スクレイピング直後のデータでは文字列のまま保持されるため、この型不一致で
  騎手・調教師の統計がほぼ全件マッチせずNaNになっていました(CSV往復した
  データ同士を比較していたテスト・デモでは症状が出ず、`predict_raceday.py`
  でライブ出馬表と結合して初めて発覚)。`keiba_ai.io.read_race_csv`で
  ID列を明示的に文字列型として読み込むよう修正し、全CSV読み込み箇所を
  これに統一しました(`tests/test_io.py`で回帰テスト済み)。

## 自動化(実運用に向けて)

```bash
# 差分更新: 前回の最終日の翌日〜昨日までを自動スクレイピングし、
# データセットに追記して再学習まで一括で行う
python scripts/update_dataset.py \
    --data data/jra_results.csv \
    --model-out models/model_dirt.joblib --history-out models/history.csv \
    --contact your-email@example.com

# 開催日一括予測: 指定日の中央競馬(ダート)レースを自動検出し、
# 出馬表を取得してレースごとに予測をまとめて表示
python scripts/predict_raceday.py --date 20250111 --dirt-only \
    --model models/model_dirt.joblib --history models/history.csv \
    --contact your-email@example.com --out reports/20250111.csv
```

- `update_dataset.py` は `--data` の最終日を見て次の日から自動的に日付範囲を
  計算するので、日付を手計算する必要はありません。データが存在しない状態
  (初回)では `--start-date` を渡してください。
- `predict_raceday.py` は `race.netkeiba.com` のライブ開催スケジュール
  (レース結果データベースではなく、まだ結果の出ていない開催カード)から
  その日のレースを自動検出します。レース開催の数日前〜当日に、開催者が
  カードを公開してから使えます(それより先の日付は未公開のため空になります)。

### GitHub Actions での自動更新

`.github/workflows/weekly_update.yml` が毎週月曜18:00 UTC(火曜3:00 JST、
週末のJRA開催が一通り終わった後)に `update_dataset.py` を実行します。

- 使うには、リポジトリの Settings → Secrets and variables → Actions →
  Variables で **`SCRAPER_CONTACT_EMAIL`**(スクレイパーのUser-Agentに
  埋め込む連絡先)を設定してください。
- 更新された `data/jra_results.csv` は自動的にコミット・pushされます。
- `models/model_dirt.joblib` / `models/history.csv` はリポジトリにコミット
  せず(`data/jra_results.csv` から数秒で再現できるため)、ワークフローの
  実行結果(Artifacts)からダウンロードする形にしています。
- 手動実行(`workflow_dispatch`)にも対応しているので、Actionsタブから
  いつでも即座にトリガーできます。

⚠️ 定期的な自動スクレイピングになるため、有効化する前に対象サイトの利用規約を
再度ご確認ください。

## Streamlit UIのモード

- **デモデータで試す**: 合成データでその場学習・予測(ネットワーク不要)。
- **実データモデルを使う(学習済み)**: `models/model_dirt.joblib` /
  `models/history.csv` を読み込み、再学習せずに実データでの予測を確認。
  履歴データ内のダートレースを選んで予測 vs 実際の着順を見比べられます。
  事前に上記の収集・学習コマンドを実行しておく必要があります。
- **今日・明日のレースを予想**: `predict_raceday.py`と同じ仕組みをUIに統合し、
  コマンドラインを使わずにこれから開催されるレースを予想できるモード。
  日付(今日/明日ボタンまたはカレンダー)を選ぶと、その日の中央競馬の
  開催カードを`race.netkeiba.com`から取得し、レースごとに予想結果を表示する。
  出馬表がまだ発表されていない日付は空振りになる(通常、開催の数日前に発表)。
- **CSVをアップロード**: 自前の過去成績CSV・出馬表CSVで学習・予測。
- **URLから取得(スクレイピング)**: netkeibaのURLをその場で取得して学習・予測
  (少数URLでは精度は出ません。デモ用途)。

## テスト

```bash
pytest tests/ -v
```

`tests/fixtures/` の静的HTMLを使って `parser.py`/`scraper.py`(開催日一覧の
場ごとグルーピング含む)を検証し、`features.py` は手作りの小さなデータ
セットでリークが起きていないこと(ダート限定履歴・コースバイアス・脚質・
調教師の各特徴量について)を、`model.py` は合成データで学習したモデルが
ランダムな3頭選択(precision@3 ≈ 0.3)を明確に上回ることを確認します。
`test_io.py` は、ゼロ埋めID列がCSV往復で壊れないことを回帰テストします。
