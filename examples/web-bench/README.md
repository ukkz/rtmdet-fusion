# RTMDet-Fusion Web Bench

A small Nuxt app for checking exported RTMDet-Fusion models in the browser and comparing
runtimes and backends. It runs a model on an uploaded image or on the webcam, draws the outputs
on top of the input, and shows the inference time and frame rate.

It is a verification tool, not a production app or a reference integration: the code favors
being easy to read and switch over speed or robustness.

- Runtimes: TensorFlow.js and ONNX Runtime Web
- Backends: WebGPU, WebGL (TensorFlow.js only; the default `onnxruntime-web` bundle has no WebGL backend)
  and WASM. Unavailable backends fall back (tfjs to WebGL, ONNX to WASM).
- Overlays: person boxes and masks, head (11) and whole-body (31) keypoints, part masks, depth

The model's inputs and outputs are described in [docs/output-contract.md](../../docs/output-contract.md).

## Prerequisites

- Node.js `^20.19.0` or `>=22.12.0` and pnpm
- macOS on Apple Silicon. The app is only tested there, and `package.json` lists darwin-arm64
  builds of native packages (`@esbuild/darwin-arm64`, `@rollup/rollup-darwin-arm64`,
  `lightningcss-darwin-arm64`, ...) as devDependencies, so `pnpm install` is expected to work
  as-is only on that platform. On other platforms, remove these platform-specific devDependencies
  or replace them with the packages for your platform before installing.
- A browser with WebGPU for the WebGPU backend (e.g. Chrome 113+). WebGL and WASM work in any modern browser.
- Exported models (see below). The app does not ship with any model.

## Getting the models

Both runtimes load models from `public/`:

| Runtime | Location | Produced by |
|---|---|---|
| ONNX Runtime Web | `public/onnx_model/rtmdet_fusion_{variant}_{size}_{version}.onnx` | `tools/onnx_exporter.py` |
| TensorFlow.js | `public/tfjs_model/{variant}_{size}_{version}/model.json` | `tools/tf-converter` + `tools/tfjs-converter` |

The models are not tracked by git. `public/onnx_model` is a tracked symlink to `onnx_models/` at
the repository root, which is ignored, and `public/tfjs_model/` is ignored. The other tracked
files in `public/` are `favicon.ico` and `robots.txt`.

1. Export ONNX models with `tools/onnx_exporter.py` and put them into `onnx_models/`
   (file names `rtmdet_fusion_{variant}_{size}_{version}.onnx`, e.g. `rtmdet_fusion_s_640_v2.0.0.onnx`).
2. Convert them to TensorFlow.js with `tools/tf-converter` (ONNX to SavedModel) and then
   `tools/tfjs-converter` (SavedModel to TensorFlow.js). The latter writes directly into
   `public/tfjs_model/`. See [tools/tf-converter/README.md](../../tools/tf-converter/README.md).

Step-by-step instructions are in [docs/export.md](../../docs/export.md).

The model version and the list of selectable models are set at the top of
`app/composables/useRtmdetModel.ts` (`MODEL_VERSION`, `AVAILABLE_MODELS`; default: `s_640`,
`tiny_256` and `tiny_640` of `v2.0.0`). Edit `AVAILABLE_MODELS` to match the models you exported.
To compare the two runtimes, each listed model needs both the ONNX file and the TensorFlow.js model.

## Running

```bash
cd examples/web-bench
pnpm install            # once; pnpm may ask you to approve dependency build scripts
node_modules/.bin/nuxi dev
```

Then open http://localhost:3000.

Start the dev server with `node_modules/.bin/nuxi dev` rather than `pnpm dev`: recent pnpm
versions run a dependency check before scripts, which stops on unapproved build scripts.

A sample image is not included. Optionally, place a photo with people at `public/test.jpg`
(ignored by git); `tools/tf-converter` uses it as the representative input during conversion.

## Using the app

1. **View**: what to overlay
   - *Keypoints*: person boxes, instance masks, whole-body skeleton (points predicted to be out of
     frame are hidden), head box and head skeleton
   - *Part masks*: head, torso, limbs and hands
   - *Depth*: per-person depth inside the body mask (red = near, blue = far)
2. **Runtime** and **Backend**: TensorFlow.js on WebGPU, WebGL or WASM, or ONNX Runtime Web on WebGPU or WASM.
   The status line shows the backend that is actually in use after any fallback.
3. **Model Variant** / **Model Resolution**: only the models in `AVAILABLE_MODELS` can be selected.
4. **Start camera** runs inference continuously on the webcam (shown mirrored) and reports
   ms per frame and FPS. **Choose File** runs inference once on an image and reports the inference time.

Both runtimes get the same input: the image at its original resolution (or the camera frame)
stretched to S x S on a canvas, regardless of the size at which it is displayed.

The reported time covers preprocessing, inference and reading the outputs back to JavaScript
(not drawing). The first run after loading a model is done during loading as a warm-up.

All detections in `valid_count` are drawn. Exported models keep detections with a score of
0.05 or more, so low-confidence boxes can appear; an application should filter by `scores`.

## Checks

```bash
node_modules/.bin/nuxi typecheck
node_modules/.bin/eslint .
node_modules/.bin/nuxi build
```

`nuxi typecheck` needs `vue-tsc`, which is not a dependency, and fetches it with `npx`. If that
picks up TypeScript 7 and fails (`Package subpath './lib/tsc' is not defined`), run vue-tsc with
TypeScript 5.9 from a scratch directory instead:

```bash
npm install --prefix /tmp/vue-tsc vue-tsc@3.2 typescript@5.9
/tmp/vue-tsc/node_modules/.bin/vue-tsc -b --noEmit
```

`nuxi build` copies `public/`, including the models, into `.output/` (about 1 GB with a few
models); delete `.output/` afterwards if you only used the build as a check.
