# 競馬予想AI (keiba-ai)

過去のレース結果から「複勝圏内(3着以内)に入る確率」を予測する、学習用の
最小構成AIパイプラインです。データ収集(スクレイピング)・特徴量エンジニアリング・
LightGBMモデル・Streamlit UI までを一通り含みます。

**⚠️ 重要な注意事項**
- 実サイトへのスクレイピング機能を含みますが、**対象サイトの利用規約(ToS)・
  robots.txt を必ず確認し、遵守できる範囲でのみ使用してください。** 本リポジトリの
  スクレイパーは既定でリクエスト間隔を空け、robots.txt が禁止しているURLへの
  アクセスは拒否しますが、これは免責にはなりません。利用は自己責任でお願いします。
- 予測結果は娯楽・研究目的のものであり、**賭け金や馬券購入の助言ではありません。**
- 同梱の合成データ (`synth_data.py`) は実在のレース・馬・騎手とは無関係の
  架空データです。ネットワークなしでパイプライン全体を試すために用意しています。

## 構成

```
src/keiba_ai/
  parser.py      # HTMLパース(純粋関数、ネットワーク非依存)
  scraper.py      # robots.txt尊重・レート制限付きのネットワーク層
  synth_data.py   # オフラインデモ/テスト用の合成レースデータ生成
  features.py     # 未来情報リークなしの特徴量エンジニアリング
  model.py        # LightGBMモデルの学習・保存・推論
  app.py          # Streamlit UI
scripts/
  generate_demo_data.py       # 合成データをCSVに書き出す
  scrape_jra_dirt_results.py  # 中央競馬(JRA10場)の実レース結果を収集
  train_model.py               # CSVからモデルを学習(--dirt-onlyでダート特化)
  predict_race.py              # 出馬表(CSV/URL)から予測
tests/                          # pytest (parser/features/modelのユニットテスト)
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
限定成績)特徴量が十分に貯まらず精度が伸び悩みます。実際に検証したところ:

| 収集期間 | ダートレース数 | held-out AUC |
|---|---|---|
| 約3週間(2024年1月) | 133 | 0.574 |
| 半年(2024年1〜6月) | 893 | 0.722 |
| **1年(2024年1〜12月)** | **1,660** | **0.738** |

収集期間を延ばすほど、特に `horse_dirt_avg_rank_before`(馬のダート平均着順)
のような馬側のダート適性特徴量が効いてきます。1年分では
`course_waku_bias_*`(コース×距離帯×枠のバイアス、下記参照)も特徴量重要度の
上位に入り、コース・枠順による構造的な有利不利も捉えられるようになりました。

⚠️ 実サイトへのアクセスを行うため、事前に対象サイトの利用規約を確認してください。

## モデルについて

- 目的変数: `target_top3` (3着以内なら1、それ以外は0) の二値分類 (LightGBM)。
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
  傾向を捉えるための特徴量です。1年分のデータで学習したところ、この
  特徴量群は特徴量重要度の上位に入り、実際に予測に寄与していることを
  確認しました。
- 検証は `GroupShuffleSplit` でレース単位に分割し(同一レースの馬が
  学習/検証に分かれて跨がらないように)、held-outレースでのAUCを表示します。

## Streamlit UIのモード

- **デモデータで試す**: 合成データでその場学習・予測(ネットワーク不要)。
- **実データモデルを使う(学習済み)**: `models/model_dirt.joblib` /
  `models/history.csv` を読み込み、再学習せずに実データでの予測を確認。
  履歴データ内のダートレースを選んで予測 vs 実際の着順を見比べられます。
  事前に上記の収集・学習コマンドを実行しておく必要があります。
- **CSVをアップロード**: 自前の過去成績CSV・出馬表CSVで学習・予測。
- **URLから取得(スクレイピング)**: netkeibaのURLをその場で取得して学習・予測
  (少数URLでは精度は出ません。デモ用途)。

## テスト

```bash
pytest tests/ -v
```

`tests/fixtures/` の静的HTMLを使って `parser.py` を検証し、`features.py` は
手作りの小さなデータセットでリークが起きていないことを、`model.py` は
合成データで学習したモデルがランダム予測(AUC 0.5)を明確に上回ることを確認します。
