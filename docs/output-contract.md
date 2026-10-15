# 入出力仕様

書き出したモデル（ONNX/CoreML/TensorFlow.js/TFLite）をアプリケーションへ組み込むための仕様です。
出力名・並び順・メタデータキーの実装上の正は`rtmdet_fusion/utils/export_contract.py`（全エクスポータ共通）です。
書き出し手順は[export.md](export.md)にあります。

## 1. 入力

| 形式 | 名前 | 形状 | 内容 |
|---|---|---|---|
| ONNX / CoreML | `images` | (1, 3, S, S) | **BGR**・0〜255・未正規化（正規化はモデル内で行う） |
| TensorFlow / tfjs / TFLite | `images` | (1, S, S, 3) | 同上（チャネル最後） |

CoreMLは入力・出力ともFloat16です（ONNX・TensorFlow系はfloat32、`valid_count`のみint32）。

Sは書き出し時の解像度（256 / 320 / 416 / 512 / 640）です。元画像は**縦横比を保たずS×Sへ引き伸ばします**
（学習時と同じ非等方リサイズ）。出力座標はこのS×S空間なので、元画像へはxを`W/S`、yを`H/S`倍して戻します。

## 2. 出力（19本）

M = 出力枠数（既定5）です。有効なのは先頭`valid_count`枠で、検出スコアの降順に並びます。

| # | 名前 | 形状 | 内容 |
|---|---|---|---|
| 0 | `bboxes` | (M, 4) | 人物bbox [x1, y1, x2, y2]（S×S空間、画像内にクリップ） |
| 1 | `scores` | (M,) | 検出スコア |
| 2 | `masks` | (M, S, S) | 人物インスタンスマスク（見えている領域。logitを平滑化した0〜1の値で、0.5がlogit 0） |
| 3 | `head_bboxes` | (M, 4) | 頭部の参照bbox。`keypoints`の基準。画像外・負座標を取りうる |
| 4 | `keypoints` | (M, 11, 3) | 頭部11点[rel_x, rel_y, rel_z]（§3） |
| 5 | `confidences` | (M,) | 頭部11点の平均信頼度[0, 1] |
| 6 | `visibilities` | (M, 11) | 頭部各点の可視確率 |
| 7 | `valid_count` | (1,) | 有効枠数（ONNX・TensorFlow系はint32、CoreMLはFloat16） |
| 8 | `head_quaternions` | (M, 4) | 頭部姿勢[w, x, y, z]（§4） |
| 9 | `appearance_features` | (M, 256) | 頭部領域の特徴量の平均（L2正規化） |
| 10 | `pose_features` | (M, 256) | 頭部キーポイント特徴量の平均（L2正規化） |
| 11 | `in_frame` | (M, 11) | 頭部各点の画面内確率 |
| 12 | `body_bboxes` | (M, 4) | 全身の参照bbox。`body_keypoints`の基準。画像外を取りうる |
| 13 | `body_keypoints` | (M, 31, 3) | 全身31点[rel_x, rel_y, rel_z]（§3・§5） |
| 14 | `body_confidences` | (M, 31) | 全身各点の信頼度（頭部と違い点ごと） |
| 15 | `body_visibilities` | (M, 31) | 全身各点の可視確率 |
| 16 | `body_in_frame` | (M, 31) | 全身各点の画面内確率 |
| 17 | `part_masks` | (M, 7, S/4, S/4) | 部位マスク確率（§6）。**TensorFlow / tfjs / TFLiteでは(M, S/4, S/4, 7)** |
| 18 | `depth` | (M, S/4, S/4) | 深度（§7） |

- 無効枠（`valid_count`番目以降）は`keypoints` / `body_keypoints`以外がすべて0です（bboxも0）。キーポイントの相対座標には意味のない値が入るので、必ず`valid_count`で判定してください
- CoreMLの出力名は`out_` + 名前です（`confidences`だけは`out_kpt_confidences`）
- tfjsは`model.predict()`が上表の名前をキーにしたobjectを返します
- `appearance_features` / `pose_features`は識別用の学習（metric learning等）をしていない中間特徴量で、
  人物の再識別にそのまま使える精度はありません

## 3. キーポイント座標（頭部・全身共通）

参照bbox [x1, y1, x2, y2]（頭部は`head_bboxes`、全身は`body_bboxes`）に対する相対値です。

```
img_x = rel_x * (x2 - x1) + x1
img_y = (1 - rel_z) * (y2 - y1) + y1      # rel_z は上が 1
奥行き = (rel_y - 0.5) * ((x2 - x1) + (y2 - y1)) / 2   # S×S 空間の画素等価、正 = 奥
                                                         # 基準点 = 頭部は center、全身は pelvis
```

- rel_x / rel_zは[-0.125, 1.125]を取りえます（参照bboxの外・画面外の点を表現するため）。[0, 1]にクランプしないでください
- 画面外の点は`in_frame` / `body_in_frame` < 0.5で判別します（座標自体は推定値が出ます）

### 頭部11点

| idx | 名前 | idx | 名前 |
|---|---|---|---|
| 0 | center（頭部中心、奥行きの基準点） | 6 | nose |
| 1 | top（頭頂） | 7 | chin |
| 2 | left_ear | 8 | occiput（後頭部） |
| 3 | right_ear | 9 | vertex（頭頂、別アングル用） |
| 4 | left_eye | 10 | neck |
| 5 | right_eye | | |

## 4. 頭部姿勢（`head_quaternions`）

頭部キーポイント（center・top・nose）から求めた回転を[w, x, y, z]（正規化済み）で表します。

- 座標系: X = 画像右、Y = 奥行き（奥が正）、Z = 画像上
- 基準姿勢（q = [1, 0, 0, 0]）: 正面方向（center→nose）が−Y（カメラを向く）、上方向（center→top）が+Z

## 5. 全身31点（先頭17点はCOCO-17と同順）

| idx | 名前 | idx | 名前 | idx | 名前 |
|---|---|---|---|---|---|
| 0 | nose | 11 | left_hip | 22 | right_collar |
| 1 | left_eye | 12 | right_hip | 23 | left_big_toe |
| 2 | right_eye | 13 | left_knee | 24 | right_big_toe |
| 3 | left_ear | 14 | right_knee | 25 | left_heel |
| 4 | right_ear | 15 | left_ankle | 26 | right_heel |
| 5 | left_shoulder | 16 | right_ankle | 27 | left_thumb_tip |
| 6 | right_shoulder | 17 | pelvis（奥行きの基準点） | 28 | right_thumb_tip |
| 7 | left_elbow | 18 | spine2 | 29 | left_middle_tip |
| 8 | right_elbow | 19 | neck | 30 | right_middle_tip |
| 9 | left_wrist | 20 | head | | |
| 10 | right_wrist | 21 | left_collar | | |

骨格の結線は`configs/default.py`の`BODY_SKELETON`です。

## 6. 部位マスク（`part_masks`）

| ch | 内容 |
|---|---|
| 0 | 着衣を含む全身（他の物体に隠れた部分も含むamodal） |
| 1 | 素体の全身（amodal） |
| 2 | 頭 |
| 3 | 胴 |
| 4 | 四肢 |
| 5 | 手 |
| 6 | 遮蔽領域（ch0のうち他の人物・物体に隠れている部分） |

- 値は確率（sigmoid済み）です。表示は0.5で2値化するのが目安です
- 解像度はS/4（640 → 160×160）で、S×Sの全面に対応します。元画像へは引き伸ばして重ねます
- `masks`とは別物です: `masks`は見えている人物領域、`part_masks` ch0は隠れた部分も含みます。
  見えている領域はch0 ∧ ¬ch6、髪・衣服の領域はch0 − ch1として導出できます

## 7. 深度（`depth`）

- 全身キーポイントのrel_yと同じ表現です（0.5 = 骨盤、`body_bboxes`の(w + h) / 2で正規化）。
  §3の式で画素等価の奥行きへ変換できます（正 = 奥）
- 意味のある値は`part_masks` ch0 > 0.5の画素だけです（それ以外は不定）
- 表示例（`tools/viewer.py` / web-bench）: 人物ごとに0.5を中心とした対称範囲で正規化し、手前を赤・奥を青で描きます

## 8. 成果物のメタデータ

版と書き出し設定を成果物自体に記録しています（足切り等だけを変えて再出力した成果物を区別するため）。
埋め込み先はONNX = `metadata_props`、CoreML = `user_defined_metadata`、tfjs = `model.json`の`userDefinedMetadata`です。

| キー | 例 | 内容 |
|---|---|---|
| `rtmdet_fusion_version` | `v2.0.0` | 版（重みを生んだコードのgit tag） |
| `rtmdet_fusion_score_threshold` | `0.05` | 検出スコアの足切り |
| `rtmdet_fusion_iou_threshold` | `0.6`（ONNX / tfjs）/ `0.7`（CoreML） | NMSのIoU閾値 |
| `rtmdet_fusion_max_instances` | `5` | 出力枠数 |
| `rtmdet_fusion_topk` | `100` | NMS前の候補数 |
| `rtmdet_fusion_builtin_nms` | `true` | NMSをモデルに内蔵しているか |
| `rtmdet_fusion_grid_sample` | `matmul` | bbox特徴の標本化方式 |

検出スコアの足切りは既定で0.05と低く焼いてあります（`valid_count`には0.05以上の検出が入ります）。
アプリ側で`scores`を見て用途に合った閾値で絞ってください。
