# データ準備（AGORA）

学習データは、合成データセット[AGORA](https://agora.is.tue.mpg.de)の画像・人物マスクとSMPL-Xのフィットから
`tools/agora/`で生成します。**AGORA・SMPL-Xとも本リポジトリには含まれません。各自で利用登録し、ライセンス
（非商用・再配布禁止）に同意して入手してください。** 生成したデータも同じライセンスに従います。

## 1. 用意するもの

| もの | 入手先 | 置き場所 |
|---|---|---|
| AGORAの画像（4Kまたは1280×720）・人物マスク（train / validation） | [AGORA](https://agora.is.tue.mpg.de) | `$AGORA_DIR/{split}/images4K/`・`masks4K/`（720pは`images/`・`masks/`） |
| AGORAのSMPL-X正解に2D / 3D関節と頂点を加えたDataFrame（`{split}_*_withjv.pkl`） | AGORAのSMPL-X正解から、[agora_evaluation](https://github.com/pixelite1201/agora_evaluation)の関節投影の手順（`project_joints`）で生成します（解像度は既定の3840×2160のまま。`--resolution 720p`で抽出する場合も同じ） | `$AGORA_DIR/{split}/SMPLX/` |
| SMPL-Xモデル（`SMPLX_NEUTRAL.npz`） | [SMPL-X](https://smpl-x.is.tue.mpg.de) | `weights/smplx/` |
| SMPL-Xの頂点の部位分割（`smplx_vert_segmentation.json`） | Meshcapadeが公開しているもの | `weights/smplx/` |
| RTMDet-Insの事前学習重み（`rtmdet-ins_s.pth`） | [setup.md](setup.md) | `weights/` |

- `AGORA_DIR`はAGORAのルートです。環境変数で指定するか、各ツールの`--agora-dir`で渡します（既定は`<repo>/datasets/agora`）
- trainの画像は配布時の分割ごとに`0/`〜`9/`のサブフォルダに置いても構いません
- マスクはシーンごとのフォルダに、人物ごとのマスクと全員分の統合マスク（`_00000`）がある前提です
- `withjv.pkl`で使う列: `imgPath`、`isValid`、`gt_joints_2d`、`gt_joints_3d`、`gt_verts`（あれば`gt_path_smplx`・`kid`・`occlusion`）
- SMPL-Xのアセットが無いと、部位マスクと深度の教師は空になります
- `tools/agora`はpklの読み込みにpandas 2.xを使います（`pyproject.toml`には含めていないので`uv pip install "pandas<3"`。pandas 3.xには未対応です）

```
$AGORA_DIR/
├── train/
│   ├── images4K/{0..9}/*.png
│   ├── masks4K/<scene>/*_mask_*.png
│   └── SMPLX/train_*_withjv.pkl
└── validation/
    ├── images4K/*.png
    ├── masks4K/<scene>/*_mask_*.png
    └── SMPLX/validation_*_withjv.pkl
```

## 2. 生成

```bash
# 抽出ロジックの確認（pkl から 1 枚ずつ抽出してその場で表示する。データは書き出さない）
uv run python tools/agora/test.py --agora-dir /path/to/agora

# 生成（data/ を消してから作り直す）
uv run python tools/agora/extracter.py --agora-dir /path/to/agora --crops-per-image 10

# 生成したデータの確認
uv run python tools/agora/viewer.py
```

| オプション | 既定 | 内容 |
|---|---|---|
| `--agora-dir` | `$AGORA_DIR` | AGORAのルート |
| `--split` | `both` | `train` / `validation` / `both` |
| `--resolution` | `4K` | 元画像の解像度（`4K` / `720p`） |
| `--crops-per-image` | 1 | 元画像1枚あたりのクロップ数（v2.0.0のデータは10） |
| `--workers` | CPU数 | 並列プロセス数 |
| `--max-files` | - | splitあたりのpkl数の上限（動作確認用） |

- **`data/`は実行のたびに消して作り直します**
- 生成中に`q`を押すと（macOS・Linuxでは`q`のあとEnter）、そのsplitの処理済みの分を保存して止まります。`--split both`では次のsplitの処理に進みます
- 4Kの全量（`--crops-per-image 10`）でtrain約28万枚・44GB、validation約2万枚・3GBになります

### 出力

```
data/
├── train/images/{0..9}/<crop>.jpg, <crop>_m.jpg     クロップ画像（_m は左右反転版）
├── train/depth/{0..9}/<crop>_<k>.png                深度の教師（uint16、mm）
├── validation/images/ ・ validation/depth/
├── head_keypoints_train.json                        アノテーション（COCO 拡張形式）
└── head_keypoints_validation.json
```

## 3. 抽出の内容

- **クロップ**: 1:1・4:3・3:4・16:9・9:16の33種類（短辺480〜長辺1706）からランダムに選びます。25%の確率で
  ±5〜55°回転します。各クロップは左右反転版も保存します（左右のキーポイントは入れ替えます）
- **人物の対応付け**: 各クロップでRTMDet-Insを実行し、AGORAのマスクとハンガリアン法で対応付けます
  （IoU ≥ 0.3、またはcoverage = 交差 / 検出マスク ≥ 0.7かつ複合スコア ≥ 0.4）。誰とも対応しないクロップはやり直します
- **キーポイント**: SMPL-Xの関節・メッシュ頂点から求めます（下表）。AGORAは画面外の点も正しい座標を持つので、
  画面外の点も座標を残しvisibility = 0にします
- **visibility**: 0 = 画面外、1 = 遮蔽（より手前の他人のマスク内）、2 = 可視
- **ignore**: 頭部bboxが小さい（面積 < 1000px²）か頭部が完全に画面外なら頭部系をignore、全身参照bboxが小さい
  （< 3000px²）か全身が画面外なら全身系をignoreします（`body_ignore`）。画像外へはみ出すbboxは許容します
- **教師マップ**: 部位マスク（頭・胴・四肢・手）と深度はSMPL-Xの人物単独レンダリングから作ります（クロップの1/4解像度）。
  遮蔽領域はAGORAの人物マスク（隠れた部分も含む）と統合マスク（見えている部分）の差から作ります（クロップと同じ解像度）
- **焦点距離**: シーンごとのレンズ（4K基準でarchviz 1920px、hdri 5333.3px、その他2986.7px）を使い、擬似深度の換算に使います

### 頭部11点の求め方

| idx | 名前 | SMPL-Xからの取得方法 |
|:--:|---|---|
| 0 | center | 関節15（頭部中心。頭部の奥行きの基準点） |
| 1 | top | neck → centerの延長線に最も近いメッシュ頂点 |
| 2 / 3 | left_ear / right_ear | 関節59 / 58 |
| 4 / 5 | left_eye / right_eye | 関節57 / 56 |
| 6 | nose | 関節55 |
| 7 | chin | nose–neck線分の1/4〜3/4区間に最も近いメッシュ頂点 |
| 8 | occiput | nose → centerの延長点とcenterを結ぶ線分に最も近いメッシュ頂点 |
| 9 | vertex | center→topとcenter→occiputの合成ベクトルの延長線に最も近いメッシュ頂点 |
| 10 | neck | 関節12 |

全身31点はSMPL-Xの関節をそのまま使います（対応は`configs/default.py`の`BODY_KEYPOINT_DEFS`）。

## 4. アノテーション形式

COCO形式を拡張したものです。

`images`の主な項目: `id`、`file_name`、`width`、`height`、`focal_length`（クロップの解像度に換算済み）、
`rotated`（回転角）、`mirrored`、`cx` / `cy`（元画像の画素座標でのクロップの中心）

`annotations`の主な項目:

| 項目 | 内容 |
|---|---|
| `bbox`・`segmentation`・`area` | 人物bboxと人物マスク（RLE、隠れた部分も含む） |
| `head_bbox` | 頭部bbox（画面外を含む11点の外接矩形を正方形化して1.2倍。画像外へはみ出せます） |
| `keypoints` / `keypoints_cam` | 頭部11点。2D（x, y, v）/ 統一座標（x, y, z, v） |
| `keypoints_body` / `keypoints_body_cam` | 全身31点。同上 |
| `num_keypoints` / `num_body_keypoints` | v = 2の点の数 |
| `body_ref_bbox` | 全身31点の外接正方形 × 1.2（学習では使わず、データセットがキーポイントから矩形を作り直します） |
| `ignore` / `body_ignore` | 頭部系 / 全身系を損失から外すか |
| `pelvis_depth` | 骨盤のカメラ深度（m） |
| `part_masks`・`occlusion_mask`・`occlusion_valid` | 部位マスク・遮蔽領域の教師（RLE）と、遮蔽の教師が信頼できるか |
| `depth_file` | 深度の教師PNG（左右反転版は元画像と同じファイルを使い、データセットが反転して読みます） |
| `subject_id`・`kid`・`occlusion_pct` | スキャンのID、子どもか、遮蔽率 |
| `matched_ins_id` | 対応付いたRTMDet-Insの検出 |

統一座標（`*_cam`）:

- `x`: 画像のX座標（右が正）
- `y`: 擬似深度（画素等価のスケール。基準点 = 頭部はcenter、全身はpelvisが0）
- `z`: 画像のY座標を反転したもの（上が正、`z = H - image_y`）
- `v`: visibility（0 / 1 / 2）

学習時はデータセットがこれを参照bboxに対する相対座標（[output-contract.md](output-contract.md) §3）に変換します。
