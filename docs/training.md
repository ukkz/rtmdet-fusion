# 学習

`tools/train.py`で学習します。データの用意は[data-preparation.md](data-preparation.md)、構造は[architecture.md](architecture.md)を参照してください。
RTMDet-Insは凍結したまま、Hybrid Encoder以降のヘッド群だけを学習します。

## 1. 版を付けてから学習する

学習したモデルの版はgit tag（SemVer）で表し、学習開始時の`git describe --tags --dirty`がチェックポイントに
埋め込まれます。**学習するコミットに先にtagを打ってから学習してください**（ルールの正は[AGENTS.md](../AGENTS.md)「Versioning」）。

```bash
git tag -a v2.1.0 -m "..."      # 学習するコミットに打つ（採用が決まるまで push しない）
uv run python tools/train.py ...
```

- `train.py`は、作業ツリーがクリーンかつHEADがちょうどtagの位置でなければ起動しません
- 試行錯誤の学習には`--experiment`を付けてください。チェックポイントには`v2.0.0-3-gabc1234`のような値が入り、リリースではないことが分かります
- 採用したらtagをpushしてください。採用しなければtagとチェックポイントを消し、同じ番号を使い直して構いません
- 書き出した成果物はチェックポイントの版を引き継ぎます（[export.md](export.md)）

## 2. 2段階の学習

マスク・深度ブランチは、キーポイントの学習が済んだ後に追加して学習します。

```bash
# stage 1: 検出・頭部・全身キーポイント（マスク・深度ブランチなし）
uv run python tools/train.py --amp --epochs 40

# stage 2: stage 1 の最良チェックポイントから再開し、マスク・深度ブランチを追加して学習
uv run python tools/train.py --amp --resume best --mask-weight 4.0 --epochs 50
```

- `--epochs`は通しの総エポック数です。`--resume`はチェックポイントのエポックの次から`--epochs`まで続け、
  学習率スケジュールも再開位置まで進めます（上の例の40 / 50は一例です）
- stage 2ではoptimizerの状態を引き継がず、EMAは既存モジュールの分をチェックポイントから引き継ぎます（追加したブランチは初期の重みから始まります）。最良val lossもリセットされます
  （マスク・深度の損失が加わり、stage 1の値と比べられないため）
- stage 2の最初のエポックで同じ`--work-dir`の`best_model_{variant}.pth`が上書きされます。stage 1の結果を残したい場合は、
  事前に別名でコピーしておいてください
- 書き出しにはstage 2のチェックポイントが必要です（stage 1のものはエクスポータが拒否します）。viewerはstage 1のものも表示できます
- 既定ではtinyとsを同時に学習します（`--variant both`）。片方だけなら`--variant tiny` / `--variant s`

## 3. 主なオプション

| オプション | 既定 | 内容 |
|---|---|---|
| `--epochs` / `--batch-size` / `--lr` | 100 / 48 / 5e-4 | 総エポック数・バッチサイズ・学習率（AdamW、weight decay 0.05） |
| `--variant` | `both` | `tiny` / `s` / `both` |
| `--amp` | off | 混合精度（CUDAのみ） |
| `--work-dir` | `work_dirs/main` | チェックポイントとTensorBoardログの出力先 |
| `--resume` | - | `best` / `final` / 任意名（`{name}_model_{variant}.pth`を読みます） |
| `--warmup-epochs` / `--scheduler` / `--min-lr` | 2 / `cosine` / 1e-5 | 学習率ウォームアップと減衰（`plateau`も選べます） |
| `--no-multiscale` | off | マルチスケール学習（256〜640）を止め`--img-size`（既定640）固定にします |
| `--jitter-prob` / `--jitter-shift` / `--jitter-scale` / `--no-jitter` | 0.8 / 0.25 / 0.15 | 参照bboxの摂動（§5） |
| `--no-ema` / `--ema-decay` | on / 0.9998 | 重みの指数移動平均 |
| `--body-weight` | 1.0 | 全身系の損失全体の倍率（0で全身系と、頭部を含むin_frameの損失を学習しません） |
| `--mask-weight` / `--depth-weight` | 0 / 5.0 | マスク・深度の損失の重み（`--mask-weight` > 0でブランチを作ります） |
| `--consistency-weight` / `--image-coord-weight` / `--bone-direction-weight` | 5 / 10 / 5 | 位置ずれ対策の損失（0で無効） |
| `--train-ann` / `--val-ann` / `--img-dir` / `--val-img-dir` | `data/`配下 | データの場所 |
| `--experiment` | off | tagの無いコミット・未コミットの変更での学習を許します（§1） |

全オプションは`uv run python tools/train.py --help`で確認できます。指定した引数はすべてチェックポイントの`config['train_args']`に残ります。

学習の様子はTensorBoardで見られます: `uv run tensorboard --logdir work_dirs/main/tensorboard`

## 4. 損失

頭部系と全身系に同じ構成の損失をかけます（全身系は`--body-weight`倍）。

| 損失 | 重み | 内容 |
|---|---|---|
| 参照bbox GIoU（`HeadBboxIoULoss`） | 2.0 | 頭部・全身の参照bbox |
| 参照bbox L1（`HeadBboxL1Loss`） | 12.0 | `--img-size`で正規化したL1 |
| SimCC（`SimCCLoss`） | 1.0（β = 10） | x / y / z各軸のビン分布とガウシアンターゲット（σ = 0.03）のKLダイバージェンス |
| 骨長（`BoneLoss`） | 10.0 | 骨の長さ（頭部13本・全身34本） |
| 骨方向（`BoneDirectionLoss`） | 5.0 | 骨の向き（コサイン類似度）。点群全体の平行なずれを抑えます |
| 座標L2（`CoordL2Loss`） | 15.0 | 参照bbox相対座標の直接回帰（SimCCの崩壊を防ぎます） |
| 画像座標L2（`ImageCoordL2Loss`） | 10.0 | 参照bboxの大きさで画像座標に戻したx / zのL2（予測と正解に同じ参照bboxを使います） |
| jitter一貫性（`BboxJitterConsistencyLoss`） | 5.0 | 摂動の異なる2つの参照bboxから得た画像座標を一致させます |
| visibility（`VisibilityBCELoss`） | 0.5 | 遮蔽されていないか |
| in_frame（`InFrameBCELoss`） | 0.5 | 画面内か |
| マスク（`MaskBCEDiceLoss`） | 4.0（stage 2） | 部位マスク7chのBCE + Dice（教師が無いチャネルは除外） |
| 深度（`MaskedDepthL1Loss`） | 5.0（stage 2） | 教師がある画素だけのL1 |

画面外の点（AGORAは画面外でも正しい座標を持つ）は座標の損失を重み0.3で監督し、画面内かどうかはin_frameで別に学習します。

## 5. 学習の工夫

| 手法 | 設定 | 目的 |
|---|---|---|
| ハンガリアン法によるマッチング | IoU ≥ 0.3、またはcoverage ≥ 0.7かつ複合スコア ≥ 0.4 | RTMDet-Insの検出とアノテーションを対応付けます。学習時は検出スコアの足切りを0.05まで下げ、奥の遮蔽された人物も学習に使います（検出bboxの最小サイズ100px（その時の学習解像度での値）はそのまま適用されます） |
| Scheduled Sampling | 正解bboxを使う確率 = 1 / (1 + exp(epoch − 3)) | 序盤は正解の参照bbox、以後は予測した参照bboxで特徴を取ります（推論時と同じ条件に寄せます） |
| 参照bboxの摂動（jittering） | 確率0.8、移動±25%、拡縮±15%（x / y独立） | 参照bboxの位置ではなく見た目の特徴からキーポイントを推定させます。摂動後のbboxに対する相対座標が[-0.125, 1.125]を外れた点は損失から外します |
| マルチスケール学習 | 256 / 320 / 416 / 512 / 640からランダム | 1つのモデルで書き出し解像度すべてに対応します |
| EMA | decay 0.9998 | 重みの揺れを抑えます。推論・書き出しにはEMA側の重みを使います |
| Dropout | 0.2（DCCHead） | 過学習を抑えます |

## 6. チェックポイント

`--work-dir`に`best_model_{variant}.pth`（val loss最良）と`final_model_{variant}.pth`を保存します。

```python
{
    'version': 'v2.0.0',                       # git describe --tags --dirty
    'hybrid_encoder_state_dict': ...,          # 以下、推論・書き出しに使う EMA 重み
    'bbox_feat_extractor_state_dict': ...,
    'head_bbox_head_state_dict': ...,
    'dcc_head_state_dict': ...,
    'body_bbox_head_state_dict': ...,
    'body_kpt_head_state_dict': ...,
    'mask_depth_branch_state_dict': ...,       # stage 2 のみ
    'raw_state_dicts': ...,                    # EMA 有効時の学習中の生の重み（再開用）
    'optimizer_state_dict': ..., 'epoch': ..., 'best_val_loss': ...,
    'train_metrics': ..., 'val_metrics': ...,
    'config': {'num_keypoints': 11, 'num_bins': 64, 'num_body_keypoints': 31, 'num_bins_body': 128,
               'bin_margin': 0.125, 'body_bbox_range_scale': 2.0, 'max_instances': 5, 'img_size': 640,
               ..., 'train_args': {...}},
}
```

RTMDet-Insの重みはチェックポイントに含めません（`weights/`から読みます）。バリアントは`hybrid_encoder_state_dict`の形状から判定します。

## 7. 推論の確認（viewer）

```bash
uv run python tools/viewer.py --model work_dirs/main/best_model_s.pth            # GUI で画像を選ぶ
uv run python tools/viewer.py --model work_dirs/main/best_model_s.pth --image path/to/image.jpg
```

PyQt5が必要です。表示モード（頭部・全身・部位マスク・深度など）はキー操作で切り替えます（画面のヘルプを参照してください）。
