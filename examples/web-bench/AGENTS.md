# web-bench: notes for contributors and agents

Browser verification app for exported models (TensorFlow.js on WebGPU / WebGL / WASM, ONNX Runtime
Web on WebGPU / WASM). It exists to check exported models and compare speed; it is not a
production integration. Usage is in `README.md`, the model I/O spec in `docs/output-contract.md`.

- **Start with `node_modules/.bin/nuxi dev`, not `pnpm dev` / `pnpm exec`.** Recent pnpm versions
  run a dependency check before scripts that stops on unapproved build scripts and writes a
  provisional setting into `pnpm-workspace.yaml`. Use the binaries in `node_modules/.bin` directly.
- Models (not tracked by git): tfjs in `public/tfjs_model/{variant}_{size}_{version}/`, ONNX via the
  symlink `public/onnx_model -> ../../../onnx_models` as `rtmdet_fusion_{variant}_{size}_{version}.onnx`.
  `MODEL_VERSION` / `AVAILABLE_MODELS` in `app/composables/useRtmdetModel.ts` decide what is offered.
- Input contract: **BGR**, 0-255, not normalized, stretched to S x S (no letterboxing).
  ONNX is NCHW `(1, 3, S, S)`, tfjs is NHWC `(1, S, S, 3)`. Read outputs by name, never by index.
  Both runtimes read their input from one S x S canvas (`drawInputCanvas`), so they get identical
  pixels. Do not pass an `<img>` to `tf.browser.fromPixels`: it reads the displayed (CSS) size.
- `predict()` returns null without an error while no model is set (initial load, any switch), so
  the camera loop can keep calling it. `switchBackend` unsets the model and waits for running
  inferences before the backend disposes the models / sessions. In `app.vue`, the camera loop
  ends and late results are dropped once `cameraSession` changes (stop / restart).
- `part_masks` layout differs: ONNX `(M, 7, S/4, S/4)`, tfjs `(M, S/4, S/4, 7)`. Both runtimes
  produce the same `InferenceResult`; the layout is carried in `partMaskLayout` and handled in
  `app/utils/draw.ts`. `valid_count` is int32.
- Keep tfjs models, tensors and ORT sessions outside Vue reactivity (module-level variables, and
  `shallowRef` for results). Create tensors inside `tf.tidy()` or dispose them explicitly
  (inputs and every output of `predict()`), otherwise GPU memory leaks per frame.
- `patches/@tensorflow__tfjs-backend-webgpu@4.22.0.patch` (registered in `package.json` and
  `pnpm-workspace.yaml`) fixes a WGSL type error in the WebGPU transpose kernel. Keep it when
  updating dependencies.
- Styles: Tailwind CSS v4 + Nuxt UI, imported in `app/assets/css/main.css` (`css` in
  `nuxt.config.ts`). `tailwindcss` is only a dependency of `@nuxt/ui`, which pnpm does not link
  into `node_modules/`, so `nuxt.config.ts` aliases the bare `tailwindcss` CSS import to that copy.
  If `tailwindcss` becomes a direct dependency, the alias can go.
- The WASM binaries are loaded from jsDelivr; the versions in `useRtmdetModel.ts` must match the
  installed `onnxruntime-web` and `@tensorflow/tfjs-backend-wasm`. The default `onnxruntime-web`
  bundle has no WebGL backend, so `RUNTIME_BACKENDS` offers WebGL for tfjs only.
- Before finishing a change, run the type check, `node_modules/.bin/eslint .` and
  `node_modules/.bin/nuxi build` (then delete `.output/`, which receives a copy of the models).
  `nuxi typecheck` may fail because npx resolves TypeScript 7; use the vue-tsc + TypeScript 5.9
  command in `README.md`.
