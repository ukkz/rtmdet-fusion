import type * as TF from '@tensorflow/tfjs'
import type * as ORT from 'onnxruntime-web'

// Model variants and input resolutions
export const SUPPORTED_MODEL_VARIANTS = ['s', 'tiny'] as const
export const SUPPORTED_MODEL_SIZES = [256, 320, 416, 512, 640] as const
export type ModelVariant = (typeof SUPPORTED_MODEL_VARIANTS)[number]
export type ModelSize = (typeof SUPPORTED_MODEL_SIZES)[number]

// Inference runtimes
export const SUPPORTED_RUNTIMES = ['tfjs', 'onnx'] as const
export type RuntimeType = (typeof SUPPORTED_RUNTIMES)[number]

// Backends (shared by TensorFlow.js and ONNX Runtime Web)
export const SUPPORTED_BACKENDS = ['webgpu', 'webgl', 'wasm'] as const
export type BackendType = (typeof SUPPORTED_BACKENDS)[number]

// Backends offered per runtime (the default onnxruntime-web bundle has no WebGL backend)
export const RUNTIME_BACKENDS: Record<RuntimeType, readonly BackendType[]> = {
  tfjs: ['webgpu', 'webgl', 'wasm'],
  onnx: ['webgpu', 'wasm'],
}

// Model identifier (`{variant}_{size}`)
export type ModelId = `${ModelVariant}_${ModelSize}`

// Version suffix of the model files
export const MODEL_VERSION = 'v2.0.0'
// Models that have been exported locally. Edit this list to match the files in
// public/tfjs_model/{id}_{version}/ and public/onnx_model/rtmdet_fusion_{id}_{version}.onnx
export const AVAILABLE_MODELS: ModelId[] = ['s_640', 'tiny_256', 'tiny_640']

const NUM_PART_MASK_CHANNELS = 7

// Memory layout of part_masks: tfjs returns (M, S/4, S/4, 7), ONNX returns (M, 7, S/4, S/4)
export type PartMaskLayout = 'channels-last' | 'channels-first'

// Inference result (outputs used by the drawing code, see docs/output-contract.md).
// M = number of output slots; only the first `validCount` slots are valid.
export interface InferenceResult {
  modelVariant: ModelVariant
  modelSize: ModelSize
  validCount: number
  bboxes: number[][]           // (M, 4) person bbox [x1, y1, x2, y2] in S x S space
  scores: number[]             // (M)
  masks: Float32Array          // (M, S, S) flattened
  headBboxes: number[][]       // (M, 4) reference bbox of `keypoints`
  keypoints: number[][][]      // (M, 11, 3) [rel_x, rel_y, rel_z] relative to headBboxes
  inFrame: number[][]          // (M, 11)
  bodyBboxes: number[][]       // (M, 4) reference bbox of `bodyKeypoints`
  bodyKeypoints: number[][][]  // (M, 31, 3) relative to bodyBboxes
  bodyInFrame: number[][]      // (M, 31)
  partMasks: Float32Array      // flattened, layout given by partMaskLayout
  partMaskLayout: PartMaskLayout
  depth: Float32Array          // (M, S/4, S/4) flattened
  partMaskSize: number         // S/4
}

// ============ TensorFlow.js ============

let tfInstance: typeof TF | null = null
let tfReady: Promise<typeof TF> | null = null
let currentTfjsBackend: BackendType | null = null
let isTfjsBackendSwitching = false
const tfjsModelCache = new Map<ModelId, TF.GraphModel>()

async function checkWebGPUAvailability(): Promise<boolean> {
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const adapter = await (navigator as Navigator & { gpu: GPU }).gpu.requestAdapter()
      return adapter !== null
    } catch {
      return false
    }
  }
  return false
}

async function initializeTfjsBackend(tf: typeof TF, backend: BackendType): Promise<boolean> {
  try {
    if (backend === 'webgpu') {
      await import('@tensorflow/tfjs-backend-webgpu')
      await tf.setBackend('webgpu')
      await tf.ready()
      console.log('TFJS WebGPU backend initialized')
      return true
    } else if (backend === 'webgl') {
      await import('@tensorflow/tfjs-backend-webgl')
      await tf.setBackend('webgl')
      await tf.ready()
      tf.env().set('WEBGL_PACK_DEPTHWISECONV', false)
      tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true)
      console.log('TFJS WebGL backend initialized')
      return true
    } else if (backend === 'wasm') {
      const wasmBackend = await import('@tensorflow/tfjs-backend-wasm')
      wasmBackend.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/')
      await tf.setBackend('wasm')
      await tf.ready()
      console.log('TFJS WASM backend initialized')
      return true
    }
    return false
  } catch (e) {
    console.warn(`Failed to initialize TFJS ${backend} backend:`, e)
    return false
  }
}

async function getTf(preferredBackend?: BackendType): Promise<typeof TF> {
  if (tfInstance && !preferredBackend) return tfInstance
  if (tfInstance && preferredBackend && currentTfjsBackend === preferredBackend) return tfInstance

  while (isTfjsBackendSwitching) {
    await new Promise(resolve => setTimeout(resolve, 50))
  }

  if (tfInstance && !preferredBackend) return tfInstance
  if (tfInstance && preferredBackend && currentTfjsBackend === preferredBackend) return tfInstance

  if (!tfReady || preferredBackend) {
    isTfjsBackendSwitching = true
    tfReady = (async () => {
      try {
        const tf = await import('@tensorflow/tfjs')

        if (tfInstance) {
          for (const model of tfjsModelCache.values()) {
            model.dispose()
          }
          tfjsModelCache.clear()
        }

        const backend = preferredBackend || 'webgl'
        let initialized = false

        if (backend === 'webgpu' && await checkWebGPUAvailability()) {
          initialized = await initializeTfjsBackend(tf, 'webgpu')
        } else if (backend !== 'webgpu') {
          initialized = await initializeTfjsBackend(tf, backend)
        }

        if (!initialized && backend !== 'webgl') {
          console.warn(`TFJS ${backend} backend failed, falling back to WebGL`)
          initialized = await initializeTfjsBackend(tf, 'webgl')
        }

        if (!initialized) {
          console.warn('All TFJS backends failed, falling back to CPU')
          await tf.setBackend('cpu')
          await tf.ready()
          currentTfjsBackend = null
        } else {
          currentTfjsBackend = tf.getBackend() as BackendType
        }

        tf.enableProdMode()
        console.log('TensorFlow.js initialized, backend:', tf.getBackend())

        tfInstance = tf
        return tf
      } finally {
        isTfjsBackendSwitching = false
      }
    })()
  }

  return tfReady
}

function getTfjsModelPath(modelId: ModelId): string {
  return `/tfjs_model/${modelId}_${MODEL_VERSION}/model.json`
}

// ============ ONNX Runtime Web ============

let ortInstance: typeof ORT | null = null
let ortReady: Promise<typeof ORT> | null = null
let currentOnnxBackend: BackendType | null = null
let isOnnxBackendSwitching = false
const onnxSessionCache = new Map<ModelId, ORT.InferenceSession>()

function getOnnxExecutionProvider(backend: BackendType): ORT.InferenceSession.ExecutionProviderConfig[] {
  switch (backend) {
    case 'webgpu':
      // Operators not supported by WebGPU fall back to the CPU (WASM)
      return ['webgpu', 'wasm']
    case 'wasm':
    default:
      return ['wasm']
  }
}

async function getOrt(preferredBackend?: BackendType): Promise<typeof ORT> {
  if (ortInstance && !preferredBackend) return ortInstance
  if (ortInstance && preferredBackend && currentOnnxBackend === preferredBackend) return ortInstance

  while (isOnnxBackendSwitching) {
    await new Promise(resolve => setTimeout(resolve, 50))
  }

  if (ortInstance && !preferredBackend) return ortInstance
  if (ortInstance && preferredBackend && currentOnnxBackend === preferredBackend) return ortInstance

  if (!ortReady || preferredBackend) {
    isOnnxBackendSwitching = true
    ortReady = (async () => {
      try {
        const ort = await import('onnxruntime-web')

        // Load the WASM binaries from the CDN (must match the installed version)
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/'

        // Release the existing sessions
        if (ortInstance) {
          for (const session of onnxSessionCache.values()) {
            await session.release()
          }
          onnxSessionCache.clear()
        }

        const backend: BackendType =
          preferredBackend && RUNTIME_BACKENDS.onnx.includes(preferredBackend) ? preferredBackend : 'wasm'

        // Report the backend that is actually used
        if (backend === 'webgpu') {
          const available = await checkWebGPUAvailability()
          if (!available) {
            console.warn('WebGPU not available for ONNX, falling back to WASM')
            currentOnnxBackend = 'wasm'
          } else {
            currentOnnxBackend = 'webgpu'
          }
        } else {
          currentOnnxBackend = backend
        }

        console.log('ONNX Runtime initialized, backend:', currentOnnxBackend)
        ortInstance = ort
        return ort
      } finally {
        isOnnxBackendSwitching = false
      }
    })()
  }

  return ortReady
}

function getOnnxModelPath(modelId: ModelId): string {
  return `/onnx_model/rtmdet_fusion_${modelId}_${MODEL_VERSION}.onnx`
}

// ============ Inference ============

export type ImageSource = HTMLImageElement | HTMLVideoElement

function getModelId(variant: ModelVariant, size: ModelSize): ModelId {
  return `${variant}_${size}`
}

// ============ Preprocessing (shared by both runtimes) ============

// Reusable canvas that holds the model input
let inputCanvas: HTMLCanvasElement | null = null
let inputCtx: CanvasRenderingContext2D | null = null

/**
 * Draws the source into an S x S canvas, stretched without keeping the aspect ratio (same as
 * training). drawImage uses the natural size of an image and the frame size of a video, so the
 * input does not depend on the size at which the element is displayed. Both runtimes read their
 * input from this canvas and therefore get the same pixels.
 */
function drawInputCanvas(source: ImageSource, size: ModelSize): HTMLCanvasElement {
  if (!inputCanvas || !inputCtx) {
    inputCanvas = document.createElement('canvas')
    inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true })!
  }
  if (inputCanvas.width !== size || inputCanvas.height !== size) {
    inputCanvas.width = size
    inputCanvas.height = size
  }
  inputCtx.drawImage(source, 0, 0, size, size)
  return inputCanvas
}

// TensorFlow.js inference. Input is BGR NHWC (1, S, S, 3), 0-255; outputs are read by name
async function runTfjsInference(
  tf: typeof TF,
  model: TF.GraphModel,
  canvas: HTMLCanvasElement,
  modelVariant: ModelVariant,
  modelSize: ModelSize
): Promise<InferenceResult> {
  // The canvas is already S x S; fromPixels returns RGB (alpha dropped), reversed here to BGR
  const input = tf.tidy(() => tf.reverse(tf.browser.fromPixels(canvas), -1).toFloat().expandDims(0))
  let out: TF.NamedTensorMap = {}
  try {
    // model.json's structuredOutputKeys are fixed by tools/tfjs-converter, so predict() returns named outputs
    out = model.predict(input) as TF.NamedTensorMap
    const get = (name: string): TF.Tensor => {
      const t = out[name]
      if (!t) throw new Error(`The model has no output '${name}'`)
      return t
    }
    if (get('part_masks').shape[3] !== NUM_PART_MASK_CHANNELS) {
      throw new Error(`Unexpected part_masks shape: ${get('part_masks').shape}`)
    }
    const partMaskSize = get('depth').shape[1] ?? Math.floor(modelSize / 4)

    const [bboxes, scores, validCountRaw, headBboxes, keypoints, inFrame, bodyBboxes, bodyKeypoints, bodyInFrame] =
      await Promise.all([
        get('bboxes').array() as Promise<number[][]>,
        get('scores').array() as Promise<number[]>,
        get('valid_count').array() as Promise<number[]>,
        get('head_bboxes').array() as Promise<number[][]>,
        get('keypoints').array() as Promise<number[][][]>,
        get('in_frame').array() as Promise<number[][]>,
        get('body_bboxes').array() as Promise<number[][]>,
        get('body_keypoints').array() as Promise<number[][][]>,
        get('body_in_frame').array() as Promise<number[][]>,
      ])
    const [masks, partMasks, depth] = await Promise.all([
      get('masks').data() as Promise<Float32Array>,
      get('part_masks').data() as Promise<Float32Array>,
      get('depth').data() as Promise<Float32Array>,
    ])

    return {
      modelVariant, modelSize, validCount: validCountRaw[0] ?? 0,
      bboxes, scores, masks, headBboxes, keypoints, inFrame,
      bodyBboxes, bodyKeypoints, bodyInFrame,
      partMasks, partMaskLayout: 'channels-last', depth, partMaskSize,
    }
  } finally {
    input.dispose()
    Object.values(out).forEach(t => t.dispose())
  }
}

type OnnxOutputs = ORT.InferenceSession.OnnxValueMapType

function getOnnxOutput(results: OnnxOutputs, name: string): ORT.Tensor {
  const t = results[name]
  if (!t) throw new Error(`The ONNX model has no output '${name}'`)
  return t
}

function floatData(t: ORT.Tensor): Float32Array {
  if (!(t.data instanceof Float32Array)) throw new Error(`Expected float32 output, got ${t.type}`)
  return t.data
}

// (N, C) -> number[N][C]
function toRows(t: ORT.Tensor): number[][] {
  const data = floatData(t)
  const cols = t.dims[1]!
  const rows: number[][] = []
  for (let i = 0; i < data.length; i += cols) rows.push(Array.from(data.subarray(i, i + cols)))
  return rows
}

// (N, K, D) -> number[N][K][D]
function toPoints(t: ORT.Tensor): number[][][] {
  const data = floatData(t)
  const numPoints = t.dims[1]!
  const dims = t.dims[2]!
  const perInstance = numPoints * dims
  const out: number[][][] = []
  for (let i = 0; i < data.length; i += perInstance) {
    const points: number[][] = []
    for (let k = 0; k < perInstance; k += dims) points.push(Array.from(data.subarray(i + k, i + k + dims)))
    out.push(points)
  }
  return out
}

// ONNX Runtime Web inference. Input is BGR NCHW (1, 3, S, S), 0-255; outputs are read by name
async function runOnnxInference(
  ort: typeof ORT,
  session: ORT.InferenceSession,
  canvas: HTMLCanvasElement,
  modelVariant: ModelVariant,
  modelSize: ModelSize
): Promise<InferenceResult> {
  const rgba = canvas.getContext('2d')!.getImageData(0, 0, modelSize, modelSize).data

  // RGBA (interleaved) -> BGR (planar) float32
  const area = modelSize * modelSize
  const bgr = new Float32Array(area * 3)
  for (let i = 0; i < area; i++) {
    bgr[i] = rgba[i * 4 + 2]!             // B
    bgr[i + area] = rgba[i * 4 + 1]!      // G
    bgr[i + area * 2] = rgba[i * 4]!      // R
  }

  const input = new ort.Tensor('float32', bgr, [1, 3, modelSize, modelSize])
  const results = await session.run({ images: input })
  const get = (name: string) => getOnnxOutput(results, name)

  const partMasksTensor = get('part_masks')
  if (partMasksTensor.dims[1] !== NUM_PART_MASK_CHANNELS) {
    throw new Error(`Unexpected part_masks shape: ${partMasksTensor.dims}`)
  }
  // valid_count is int32 in ONNX
  const validCountData = get('valid_count').data as Int32Array | BigInt64Array

  return {
    modelVariant, modelSize, validCount: Number(validCountData[0] ?? 0),
    bboxes: toRows(get('bboxes')),
    scores: Array.from(floatData(get('scores'))),
    masks: floatData(get('masks')),
    headBboxes: toRows(get('head_bboxes')),
    keypoints: toPoints(get('keypoints')),
    inFrame: toRows(get('in_frame')),
    bodyBboxes: toRows(get('body_bboxes')),
    bodyKeypoints: toPoints(get('body_keypoints')),
    bodyInFrame: toRows(get('body_in_frame')),
    partMasks: floatData(partMasksTensor),
    partMaskLayout: 'channels-first',
    depth: floatData(get('depth')),
    partMaskSize: partMasksTensor.dims[2]!,
  }
}

// Inferences that are still running; a backend switch waits for them before it disposes the
// models / sessions they use
const inferencesInFlight = new Set<Promise<unknown>>()

async function runInference(
  runtime: RuntimeType,
  source: ImageSource,
  modelVariant: ModelVariant,
  modelSize: ModelSize
): Promise<InferenceResult> {
  const modelId = getModelId(modelVariant, modelSize)
  if (runtime === 'tfjs') {
    const model = tfjsModelCache.get(modelId)
    if (!tfInstance || !model) throw new Error(`TFJS model ${modelId} is not loaded`)
    return runTfjsInference(tfInstance, model, drawInputCanvas(source, modelSize), modelVariant, modelSize)
  }
  const session = onnxSessionCache.get(modelId)
  if (!ortInstance || !session) throw new Error(`ONNX session ${modelId} is not loaded`)
  return runOnnxInference(ortInstance, session, drawInputCanvas(source, modelSize), modelVariant, modelSize)
}

// ============ Composable ============

export function useRtmdetModel() {
  const isLoading = ref(false)
  // Errors from loading / switching and from inference are kept apart: a successful inference
  // clears only its own kind, so a failed model load stays visible while the camera keeps running
  const loadError = ref<string | null>(null)
  const inferenceError = ref<string | null>(null)
  const error = computed(() => loadError.value ?? inferenceError.value)
  const currentRuntime = ref<RuntimeType>('tfjs')
  const currentModelVariant = ref<ModelVariant | null>(null)
  const currentModelSize = ref<ModelSize | null>(null)
  const currentBackend = ref<BackendType | null>(null)
  const isSwitchingBackend = ref(false)
  const isSwitchingRuntime = ref(false)

  const isModelReady = computed(() =>
    currentModelVariant.value !== null && currentModelSize.value !== null
  )

  const currentModelId = computed<ModelId | null>(() => {
    if (!currentModelVariant.value || !currentModelSize.value) return null
    return getModelId(currentModelVariant.value, currentModelSize.value)
  })

  function syncBackend() {
    if (currentRuntime.value === 'tfjs') {
      currentBackend.value = currentTfjsBackend
    } else {
      currentBackend.value = currentOnnxBackend
    }
  }

  function clearErrors() {
    loadError.value = null
    inferenceError.value = null
  }

  async function loadModel(variant: ModelVariant = 's', size: ModelSize = 640) {
    const modelId = getModelId(variant, size)

    if (!AVAILABLE_MODELS.includes(modelId)) {
      loadError.value = `Model ${modelId} is not available (available: ${AVAILABLE_MODELS.join(' / ')})`
      return
    }

    const cached = currentRuntime.value === 'tfjs'
      ? tfjsModelCache.has(modelId)
      : onnxSessionCache.has(modelId)
    if (cached) {
      currentModelVariant.value = variant
      currentModelSize.value = size
      clearErrors()
      return
    }

    isLoading.value = true
    clearErrors()

    try {
      if (currentRuntime.value === 'tfjs') {
        const tf = await getTf()
        syncBackend()
        const modelPath = getTfjsModelPath(modelId)

        console.log('Loading TFJS model from', modelPath)
        const model = await tf.loadGraphModel(modelPath)
        console.log('TFJS model loaded successfully:', modelId)

        tf.tidy(() => {
          const dummyInput = tf.zeros([1, size, size, 3])
          model.predict(dummyInput)
        })
        console.log('TFJS model warmup complete:', modelId)

        tfjsModelCache.set(modelId, model)
      } else {
        const ort = await getOrt()
        syncBackend()
        const modelPath = getOnnxModelPath(modelId)

        console.log('Loading ONNX model from', modelPath)
        const executionProviders = getOnnxExecutionProvider(currentOnnxBackend || 'wasm')
        const sessionOptions: ORT.InferenceSession.SessionOptions = {
          executionProviders,
          graphOptimizationLevel: 'basic',
        }
        const session = await ort.InferenceSession.create(modelPath, sessionOptions)
        console.log('ONNX model loaded successfully:', modelId)

        const dummyInput = new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size])
        await session.run({ images: dummyInput })
        console.log('ONNX model warmup complete:', modelId)

        onnxSessionCache.set(modelId, session)
      }

      currentModelVariant.value = variant
      currentModelSize.value = size
      clearErrors()
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : 'Failed to load model'
      console.error('Model loading error:', e)
    } finally {
      isLoading.value = false
    }
  }

  async function switchModel(variant: ModelVariant, size: ModelSize) {
    if (currentModelVariant.value === variant && currentModelSize.value === size) return
    await loadModel(variant, size)
  }

  async function switchBackend(backend: BackendType) {
    const currentBE = currentRuntime.value === 'tfjs' ? currentTfjsBackend : currentOnnxBackend
    if (currentBE === backend) return

    isSwitchingBackend.value = true
    isLoading.value = true
    clearErrors()

    const savedVariant = currentModelVariant.value
    const savedSize = currentModelSize.value
    // Unset the model first so that new predict() calls return null, then let running inferences
    // finish: switching the backend disposes all models / sessions of the runtime
    currentModelVariant.value = null
    currentModelSize.value = null

    try {
      await Promise.allSettled(inferencesInFlight)
      if (currentRuntime.value === 'tfjs') {
        await getTf(backend)
      } else {
        await getOrt(backend)
      }
      syncBackend()

      if (savedVariant && savedSize) {
        await loadModel(savedVariant, savedSize)
      }
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : 'Failed to switch backend'
      console.error('Backend switch error:', e)
    } finally {
      isSwitchingBackend.value = false
      isLoading.value = false
    }
  }

  async function switchRuntime(runtime: RuntimeType) {
    if (currentRuntime.value === runtime) return

    isSwitchingRuntime.value = true
    isLoading.value = true
    clearErrors()

    const savedVariant = currentModelVariant.value
    const savedSize = currentModelSize.value

    try {
      currentRuntime.value = runtime
      currentModelVariant.value = null
      currentModelSize.value = null

      // Initialize the new runtime
      if (runtime === 'tfjs') {
        await getTf()
      } else {
        await getOrt()
      }
      syncBackend()

      // Reload the model with the new runtime
      if (savedVariant && savedSize) {
        await loadModel(savedVariant, savedSize)
      }
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : 'Failed to switch runtime'
      console.error('Runtime switch error:', e)
    } finally {
      isSwitchingRuntime.value = false
      isLoading.value = false
    }
  }

  // Returns null without reporting an error while no model is set (initial load, model / backend /
  // runtime switch): callers such as the camera loop keep calling predict() during a switch
  async function predict(source: ImageSource): Promise<InferenceResult | null> {
    const variant = currentModelVariant.value
    const size = currentModelSize.value
    if (!variant || !size) return null

    const task = runInference(currentRuntime.value, source, variant, size)
    inferencesInFlight.add(task)
    try {
      const result = await task
      inferenceError.value = null
      return result
    } catch (e) {
      inferenceError.value = e instanceof Error ? e.message : 'Inference failed'
      console.error('Inference error:', e)
      return null
    } finally {
      inferencesInFlight.delete(task)
    }
  }

  return {
    isModelReady,
    isLoading: readonly(isLoading),
    error,
    currentRuntime: readonly(currentRuntime),
    currentModelVariant: readonly(currentModelVariant),
    currentModelSize: readonly(currentModelSize),
    currentModelId,
    currentBackend: readonly(currentBackend),
    isSwitchingBackend: readonly(isSwitchingBackend),
    isSwitchingRuntime: readonly(isSwitchingRuntime),
    loadModel,
    switchModel,
    switchBackend,
    switchRuntime,
    predict,
    SUPPORTED_MODEL_VARIANTS,
    SUPPORTED_MODEL_SIZES,
    SUPPORTED_BACKENDS,
    RUNTIME_BACKENDS,
    SUPPORTED_RUNTIMES,
    AVAILABLE_MODELS,
    MODEL_VERSION,
  }
}
