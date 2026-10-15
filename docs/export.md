# 書き出し（ONNX/CoreML/TensorFlow.js）

学習済みチェックポイント（stage 2まで学習したもの。[training.md](training.md)）を各ランタイム向けに書き出します。
入出力の仕様は[output-contract.md](output-contract.md)にあります。

| 形式 | ツール | 実行環境 | 主な用途 |
|---|---|---|---|
| ONNX | `tools/onnx_exporter.py` | 任意 | ONNX Runtime（Android等）、TensorFlowへの変換元 |
| CoreML | `tools/coreml_exporter_native.py` | macOSのみ | macOS / iOS |
| TensorFlow SavedModel・TFLite | `tools/tf-converter` | macOSで確認 | TFLiteはSavedModel変換の副産物 |
| TensorFlow.js | `tools/tfjs-converter` | macOSで確認 | ブラウザ |

成果物の名前とメタデータにはチェックポイントの版が入ります（例: `rtmdet_fusion_s_640_v2.0.0.onnx`、
`rtmdet_fusion_coreml_tiny_256_v2.0.0.mlpackage`）。エクスポート時のHEADではなく、重みを生んだコードの版を示します。

## 1. ONNX

```bash
# 解像度を省略すると 256 / 320 / 416 / 512 / 640 をまとめて書き出す
uv run python tools/onnx_exporter.py --checkpoint work_dirs/main/best_model_s.pth --img-size 640 --verify
```

- 出力先の既定は`onnx_models/`（git管理外。無ければ作ります）です。`--verify`はPyTorchとONNX Runtimeの出力を
  比べます（入力は乱数画像なので、検出がほとんど出ずヘッド側の確認は限られます。実画像での確認も行ってください）
- PyTorchのdynamoエクスポータでopset 18を出します。**onnxscript 0.6以上が必須**です（0.6未満は名前の衝突で不正なONNXを
  黙って書き出すため、エクスポータが起動時に止めます）
- 検出スコアの足切り（既定0.05）・NMSのIoU閾値（既定0.6）は`--score-threshold` / `--iou-threshold`で変えられます。
  値は成果物のメタデータに残ります

## 2. CoreML（macOS）

```bash
# macOS 向け（既定 target は macOS15）
uv run python tools/coreml_exporter_native.py --checkpoint work_dirs/main/best_model_s.pth --img-size 640

# iOS 向け
uv run python tools/coreml_exporter_native.py --checkpoint work_dirs/main/best_model_tiny.pth --img-size 256 \
    --deployment-target iOS17
```

- NMSをモデルに内蔵します（NMS前の候補`--topk` 100 → 出力`--max-instances` 5枠、IoU 0.7）。演算はfp16で、
  **入力と出力もすべてFloat16です**（Swift等で`MLMultiArray`を読むときは型に注意してください）
- 出力名は`out_` + 名前です（[output-contract.md](output-contract.md) §2）
- **実行はCPU+GPUで行います**（`MLComputeUnits.cpuAndGPU`）。Neural Engineでは通常の畳み込みの精度がGPUより低く、
  誤差がアテンションで増幅されてキーポイントが崩れます。`all`は解像度によってNeural Engineを選ぶことがあるので避けてください。
  CPUのみでの実行は非常に遅く、実用になりません
- `tools/coreml_compute_plan.py --mlpackage <path>`で、opごとにどの演算ユニット（ANE / GPU / CPU）へ割り当てられるかを確認できます

## 3. TensorFlow / TensorFlow.js

ONNXを`onnx_models/`に書き出してから、2つの変換ツールを順に使います（セットアップは
[tools/tf-converter/README.md](../tools/tf-converter/README.md)）。変換には人物が写った代表画像が必要です。既定の`examples/web-bench/public/test.jpg`はリポジトリに含まれていないので、
そこに置くか`--sample-image`で指定してください。

```bash
cd tools/tf-converter
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python convert_to_savedmodel.py --version v2.0.0 --variant s --size 640
cd ../tfjs-converter
.venv/bin/python convert.py --version v2.0.0 --variant s --size 640   # 既定は fp16 量子化
```

- 出力: SavedModel・TFLiteは`tools/tf-converter/output/`、tfjsは`examples/web-bench/public/tfjs_model/`
- TensorFlow系の入力はNHWC `(1, S, S, 3)`、`part_masks`は`(M, S/4, S/4, 7)`です
- onnx2tfはONNXとTFの出力を比べてレイアウト転置の向きを決めるため、人物が写った画像を代表入力に使います
  （`--sample-image`）。人物がいない画像ではキーポイントのヘッドに全ゼロが入り、誤った転置が選ばれてもわかりません
- tfjsの`model.predict()`がONNXと同じ出力名を返すよう、`convert.py`が`model.json`の出力名の並びを直します

ブラウザでの確認には[examples/web-bench](../examples/web-bench/README.md)を使ってください。

## 4. 書き出し経路の実装上の約束

推論経路はONNX（opset 18）・CoreML・onnx2tf → TensorFlow.jsのいずれにも同じグラフで変換できるように書いてあります。
書き出しに関わるコードを変えたら、ONNXの`--verify`だけでなくTensorFlow.jsまで通してPyTorchと数値を比べてください。

| 約束 | 理由 |
|---|---|
| 固定M枠の出力、テンソル演算だけの静的NMS（`NonMaxSuppression` opを使わない） | 動的形状・制御フローを避け、全ランタイムで同じグラフを動かす |
| bbox特徴の標本化は行列積（`separable_bilinear_sample`）。学習・viewerは`F.grid_sample` | 双線形補間をテント関数の重み行列の積で表す（`grid_sample`と最大7e-7で一致）。GridSample / gatherを使わず、onnx2tfの転置問題を避け、tfjsが軽く速くなる |
| convの出力と計算した座標をconcatしない（第1層の重みを分けた`bmm`の和にする） | onnx2tfのNHWC変換で2入力のレイアウトが食い違う |
| `torch.norm` / `F.normalize`を使わずsum + sqrtで書く（推論専用の関数） | opset 18のReduceL2をonnx2tfが変換できない |
| 3次元テンソルの軸1からの選択・小さな行列のstackを避け、2次元化してから成分を取る | onnx2tfが軸を取り違える |
| 部分的にブロードキャストするbatched matmulを避け、`expand`でバッチ次元を揃える | tfjsのBatchMatMulがバッチ次元を平坦化して誤計算する |
| `fusion_model_coreml.py`はCoreML向けにメインのモジュールを写したもの | 両方を同時に直す |

既知の差: onnx2tfはopset 18のResizeの`align_corners`を扱わないため、TensorFlow系ではRTMDet-Insの`masks`が
端で最大0.19ずれます（許容しています）。

## 5. 検証結果（v2.0.0、PyTorch基準の最大誤差）

`examples/web-bench/public/test.jpg`での頭部 / 全身キーポイントの最大誤差（S×S空間の画素）です。

| | s_640 | tiny_256 | tiny_640 | masks |
|---|---|---|---|---|
| ONNX Runtime | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 一致 |
| CoreML（fp16、CPU+GPU、NMS内蔵） | 0.76 / 6.3 | 0.07 / 0.87 | 0.29 / 1.95 | 0.02 |
| SavedModel | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.19（Resize） |
| TensorFlow.js（fp16、Node wasm） | 0.70 / 6.5 | 0.09 / 0.83 | 0.44 / 2.49 | 0.19（Resize） |

部位マスク・深度・in_frame等も全経路でfp16の誤差の範囲です。

参考性能（M4 Max）:

| | s_640 | tiny_256 |
|---|---|---|
| CoreML（CPU+GPU、NMS内蔵5枠） | 12.9ms | 6.0ms |
| TensorFlow.js（fp16）サイズ / Node wasm推論 | 40MB / 1.67s | 31MB / 345ms |

ブラウザのTensorFlow.js（WebGPU）ではtiny_256で約33fpsです（web-benchでの実測）。

## 6. 後処理の実装メモ

エクスポータはRTMDet-Insの後処理をテンソル演算で書き直しています。bbox・スコア・NMSの結果はMMDetectionと一致しますが、
マスクのアップサンプルは`align_corners=True`（MMDetectionは`False`）のため、マスクの輪郭は数px異なります。

- priorはoffset 0（グリッドの左上角がpriorの中心）: `shift = arange(W) * stride`
- bboxのデコードはピクセル単位の距離をそのまま使います: `x1 = shift_x - pred[..., 0]`、`x2 = shift_x + pred[..., 2]`
- マスクはprior中心からの相対座標`(prior_center - coord) / (stride * 8)`を加えたdynamic convolutionを`bmm`で計算します
- NMSはTopK（100）で候補を固定し、M回の貪欲選択でM枠を埋めます（選んだ候補自身も以後の選択から外します。検出0件でもゼロ埋めの出力で、`valid_count` = 0）
