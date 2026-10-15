# 環境構築

学習（CUDA GPU）と書き出し（CoreMLはmacOSのみ）で環境が異なります。動作を確認している構成は次のとおりです。

| 用途 | 構成 |
|---|---|
| データ準備・学習 | Windows 11 + RTX 5070 Ti（Blackwell, sm_120）+ CUDA 12.8 + PyTorch nightly（cu128）+ mmcv 2.1.0（ソースビルド）+ mmdet 3.3.0 |
| 書き出し・変換・ブラウザ確認 | macOS（Apple Silicon）+ torch 2.10.0 + torchvision 0.25.0 + coremltools 9.0 + mmcv 2.1.0（ソースビルド）+ mmdet 3.3.0 + numpy 1.26.4 |

Pythonのパッケージ管理と実行はすべて[uv](https://docs.astral.sh/uv/)で行います（`uv run python ...`）。

## 1. 学習環境（Windows + NVIDIA GPU）

`uv.lock`のtorch / torchvisionはWindows用のwheelだけを固定しています。Linuxでの学習は未検証で、CUDA 12.8以降に対応した
torch / torchvisionを自分で用意する必要があります（動作報告やPRを歓迎します）。

```bash
git clone https://github.com/ukkz/rtmdet-fusion.git
cd rtmdet-fusion

# PyTorch（pyproject.toml で CUDA 12.8 の nightly インデックスに固定）ほか
uv sync --extra viewer

# MMEngine / MMDetection（pyproject.toml の外で管理）
uv run mim install mmengine mmdet
```

### mmcv

RTMDet-Insの推論にはmmcvのCUDA拡張（NMS等）が必要です。公式のwheelは新しいPyTorch / CUDAに対応していないため、
ソースからビルドします。

- **Windows**: [scripts/mmcv-build/](../scripts/mmcv-build/README.md)のスクリプトでmmcv 2.1.0をビルドします
  （CUDA 12.8でコンパイルできない、RTMDetに不要なopを除外します）
- **Linux**（未検証）: mmcv 2.1.0を取得し、`MMCV_WITH_OPS=1`を付けて`.venv`へソースからインストールします

注意:

- `uv sync`は`pyproject.toml`に無いパッケージ（mmengine / mmdet / mmcv）を削除します。実行後は入れ直してください
- `mim install mmcv`は実行しないでください（ソースビルドしたmmcvが置き換わり動かなくなります）

### 事前学習重み（RTMDet-Ins）

OpenMMLabのCOCO学習済み重みを`weights/`に置きます（ファイル名は固定です）。

```bash
curl -L -o weights/rtmdet-ins_s.pth \
  https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth
curl -L -o weights/rtmdet-ins_tiny.pth \
  https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth
```

学習データの準備に要るSMPL-X関連のファイルとpandasは[data-preparation.md](data-preparation.md)を参照してください。

## 2. 書き出し環境（macOS）

`pyproject.toml`のtorchはCUDA版のインデックスに固定してありmacOS用のwheelが無いので、macOSでは
**uvの自動同期を止めて**手動で環境を作ります。

- `uv venv`で`.venv`を作ります
- 同梱の`.envrc`はmacOSでのみ`UV_NO_SYNC=1`を設定します。[direnv](https://direnv.net/)を使うなら
  `direnv allow`で有効にしてください。使わない場合は毎回`uv run --no-sync ...`としてください
- `.venv`に上表の版のtorch / torchvision / coremltools / numpy / scipy 1.13.1 / onnx / onnxruntime /
  onnxscript（0.6以上が必須。0.6未満はdynamo書き出しで不正なONNXを出します）を`uv pip install`で入れます
- numpyは1.26系にしてください（2.xではcoremltoolsの変換が失敗します）
- mmcv 2.1.0は`MMCV_WITH_OPS=1`でソースビルドします（setuptoolsは81未満。新しいsetuptoolsは
  `pkg_resources`が無くビルドできません）。mmengine / mmdetは`uv pip install openmim`の後に
  `uv run --no-sync mim install mmengine mmdet`で入れます
- viewerを使う場合は`uv pip install PyQt5==5.15.11`

ONNX → TensorFlow.jsの変換ツールは独立したuvプロジェクトで、別途セットアップします
（[tools/tf-converter/README.md](../tools/tf-converter/README.md)）。

## 3. MPS（Apple Silicon）での推論

viewerはMPSで動きます。mmcv / mmengineの一部がMPSに対応していないため、`rtmdet_fusion/utils/mps_workaround.py`
でNMSをCPUで行う等の回避を自動で入れています。`torch.compile()`は使えません。

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| WindowsでPyQt5とPyTorchのDLLが競合する | viewer類はtorchをPyQt5より先にimportしています。独自スクリプトでも同じ順にしてください |
| 学習中のlossがNaN | ガウシアン・KL計算はクランプ済みです。AMP（`--amp`）で起きる場合はAMPを外して切り分けてください |
