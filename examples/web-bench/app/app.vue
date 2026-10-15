<script setup lang="ts">
import type { InferenceResult, ModelSize, ModelVariant, BackendType, RuntimeType } from '~/composables/useRtmdetModel'
import { drawInferenceResult, HEAD_KEYPOINTS, PART_COLORS, VIEW_LABELS, VIEW_MODES, type ViewMode } from '~/utils/draw'

const {
  loadModel,
  switchModel,
  switchBackend,
  switchRuntime,
  predict,
  isLoading,
  error,
  currentRuntime,
  currentModelSize,
  currentModelVariant,
  currentModelId,
  currentBackend,
  isSwitchingBackend,
  isSwitchingRuntime,
  SUPPORTED_MODEL_SIZES,
  SUPPORTED_MODEL_VARIANTS,
  RUNTIME_BACKENDS,
  SUPPORTED_RUNTIMES,
  AVAILABLE_MODELS,
  MODEL_VERSION,
} = useRtmdetModel()

// Overlay mode (keypoints / part masks / depth)
const viewMode = ref<ViewMode>('kpts')

// Whether the model has been exported (see AVAILABLE_MODELS in useRtmdetModel.ts)
function isAvailable(variant: ModelVariant, size: ModelSize): boolean {
  return AVAILABLE_MODELS.includes(`${variant}_${size}`)
}

const imageRef = ref<HTMLImageElement | null>(null)
const videoRef = ref<HTMLVideoElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)

const imageUrl = ref<string | null>(null)
// Large arrays: keep them out of deep reactivity
const result = shallowRef<InferenceResult | null>(null)
const isInferring = ref(false)
const inferenceTime = ref<number | null>(null)
const selectedVariant = ref<ModelVariant>('s')
const selectedSize = ref<ModelSize>(640)

// Camera state
const isCameraMode = ref(false)
const isCameraActive = ref(false)
const cameraStream = ref<MediaStream | null>(null)
const fps = ref<number>(0)
let animationFrameId: number | null = null
let frameCount = 0
let fpsUpdateTime = 0
// Incremented by every start and stop of the camera. A camera start, inference loop or pending
// frame result that belongs to an older value is stale and is dropped.
let cameraSession = 0

const RUNTIME_LABELS: Record<RuntimeType, string> = {
  tfjs: 'TensorFlow.js',
  onnx: 'ONNX Runtime Web',
}

const BACKEND_LABELS: Record<BackendType, string> = {
  webgpu: 'WebGPU',
  webgl: 'WebGL',
  wasm: 'WASM',
}

const isBusy = computed(() => isLoading.value || isInferring.value || isSwitchingRuntime.value)

onMounted(async () => {
  await loadModel(selectedVariant.value, selectedSize.value)
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  stopCamera()
})

async function startCamera() {
  const session = ++cameraSession
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    })
    if (session !== cameraSession) {
      // Started again or stopped while waiting for the camera
      stream.getTracks().forEach((track: MediaStreamTrack) => track.stop())
      return
    }
    cameraStream.value = stream
    isCameraMode.value = true
    isCameraActive.value = true
    imageUrl.value = null
    result.value = null
    inferenceTime.value = null

    await nextTick()

    const video = videoRef.value
    if (video && session === cameraSession) {
      video.srcObject = stream
      video.onloadedmetadata = () => {
        if (session !== cameraSession) return
        video.play().catch((e: unknown) => console.warn('Failed to play the camera stream:', e))
        startInferenceLoop(session)
      }
    }
  } catch (e) {
    if (session !== cameraSession) return
    console.error('Failed to start the camera:', e)
    alert('Camera access was denied or no camera was found.')
  }
}

function stopCamera() {
  cameraSession++
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId)
    animationFrameId = null
  }

  if (cameraStream.value) {
    cameraStream.value.getTracks().forEach((track: MediaStreamTrack) => track.stop())
    cameraStream.value = null
  }

  if (videoRef.value) {
    videoRef.value.srcObject = null
  }

  isCameraActive.value = false
  isCameraMode.value = false
  fps.value = 0
  result.value = null
  inferenceTime.value = null
}

// Real-time inference loop. It ends as soon as the camera session changes (stop or restart),
// so at most one loop runs and a frame that finishes after a stop does not show its result.
function startInferenceLoop(session: number) {
  fpsUpdateTime = performance.now()
  frameCount = 0

  async function loop() {
    if (session !== cameraSession) return

    const video = videoRef.value
    if (video && video.readyState >= video.HAVE_CURRENT_DATA) {
      const startTime = performance.now()

      try {
        // null while the model is being switched (or on an error, which is shown in the status)
        const frameResult = await predict(video)
        if (session !== cameraSession) return
        result.value = frameResult
        if (frameResult) {
          inferenceTime.value = performance.now() - startTime
          frameCount++
        }
        drawOverlay()
      } catch (e) {
        console.error('Inference error:', e)
      }
    }

    const now = performance.now()
    if (now - fpsUpdateTime >= 1000) {
      fps.value = Math.round(frameCount * 1000 / (now - fpsUpdateTime))
      frameCount = 0
      fpsUpdateTime = now
    }

    animationFrameId = requestAnimationFrame(loop)
  }

  loop()
}

function handleResize() {
  if (result.value) {
    nextTick(() => drawOverlay())
  }
}

function handleFileUpload(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  const reader = new FileReader()
  reader.onload = (e) => {
    imageUrl.value = e.target?.result as string
    result.value = null
    inferenceTime.value = null
  }
  reader.readAsDataURL(file)
}

async function handleImageLoad() {
  if (!imageRef.value) return
  await runInference()
}

// Re-run inference on the loaded image (camera mode re-runs on every frame anyway)
async function rerunOnImage() {
  if (imageRef.value && imageUrl.value) {
    await runInference()
  }
}

function handleViewModeChange(mode: ViewMode) {
  viewMode.value = mode
  if (!isCameraActive.value) nextTick(() => drawOverlay())
}

async function handleRuntimeChange(runtime: RuntimeType) {
  await switchRuntime(runtime)
  await rerunOnImage()
}

async function handleBackendChange(backend: BackendType) {
  await switchBackend(backend)
  await rerunOnImage()
}

async function handleVariantChange(variant: ModelVariant) {
  selectedVariant.value = variant
  await switchModel(variant, selectedSize.value)
  await rerunOnImage()
}

async function handleSizeChange(size: ModelSize) {
  selectedSize.value = size
  await switchModel(selectedVariant.value, size)
  await rerunOnImage()
}

async function runInference() {
  if (!imageRef.value) return

  isInferring.value = true
  const startTime = performance.now()

  try {
    const imageResult = await predict(imageRef.value)
    result.value = imageResult
    inferenceTime.value = imageResult ? performance.now() - startTime : null
    nextTick(() => drawOverlay())
  } catch (e) {
    console.error('Inference error:', e)
  } finally {
    isInferring.value = false
  }
}

function drawOverlay() {
  const canvas = canvasRef.value
  const source = isCameraMode.value ? videoRef.value : imageRef.value
  if (!canvas || !source) return

  const ctx = canvas.getContext('2d')!
  // Match the canvas to the displayed size of the image / video
  const displayWidth = source.clientWidth
  const displayHeight = source.clientHeight
  canvas.width = displayWidth
  canvas.height = displayHeight
  ctx.clearRect(0, 0, displayWidth, displayHeight)
  // Without a result (e.g. during a model switch) the overlay stays empty instead of stale
  if (!result.value) return

  // The camera preview is mirrored with CSS (scaleX(-1)), so the overlay is mirrored as well
  drawInferenceResult(ctx, result.value, displayWidth, displayHeight, viewMode.value, isCameraMode.value)
}

function textColorFor(rgb: [number, number, number]): string {
  return rgb[0] + rgb[1] + rgb[2] < 400 ? 'white' : 'black'
}
</script>

<template>
  <div class="min-h-screen bg-gray-900 text-white p-4">
    <div class="max-w-6xl mx-auto">
      <h1 class="text-2xl font-bold mb-4">RTMDet-Fusion Web Bench</h1>

      <!-- Status -->
      <div class="mb-4">
        <div v-if="isSwitchingRuntime" class="text-yellow-400">
          Switching runtime...
        </div>
        <div v-else-if="isSwitchingBackend" class="text-yellow-400">
          Switching backend...
        </div>
        <div v-else-if="isLoading" class="text-yellow-400">
          Loading model ({{ selectedVariant }}_{{ selectedSize }})...
        </div>
        <div v-else-if="error" class="text-red-400">
          Error: {{ error }}
        </div>
        <div v-else class="text-green-400">
          {{ RUNTIME_LABELS[currentRuntime] }} ({{ currentModelId }}_{{ MODEL_VERSION }}) | Backend: {{ currentBackend ? BACKEND_LABELS[currentBackend] : 'N/A' }}
        </div>
      </div>

      <!-- View mode selector -->
      <div class="mb-4">
        <label class="block text-sm text-gray-400 mb-2">View:</label>
        <div class="flex gap-2">
          <button
            v-for="mode in VIEW_MODES"
            :key="mode"
            :class="[
              'px-4 py-2 rounded font-medium transition-colors',
              viewMode === mode ? 'bg-teal-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
            ]"
            @click="handleViewModeChange(mode)"
          >
            {{ VIEW_LABELS[mode] }}
          </button>
        </div>
      </div>

      <!-- Runtime selector -->
      <div class="mb-4">
        <label class="block text-sm text-gray-400 mb-2">Runtime:</label>
        <div class="flex gap-2">
          <button
            v-for="runtime in SUPPORTED_RUNTIMES"
            :key="runtime"
            :disabled="isBusy"
            :class="[
              'px-4 py-2 rounded font-medium transition-colors',
              currentRuntime === runtime
                ? 'bg-orange-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
              isBusy && 'opacity-50 cursor-not-allowed'
            ]"
            @click="handleRuntimeChange(runtime)"
          >
            {{ RUNTIME_LABELS[runtime] }}
          </button>
        </div>
        <p class="text-xs text-gray-500 mt-1">
          TensorFlow.js loads public/tfjs_model/ | ONNX Runtime Web loads public/onnx_model/
        </p>
      </div>

      <!-- Backend selector -->
      <div class="mb-4">
        <label class="block text-sm text-gray-400 mb-2">Backend:</label>
        <div class="flex gap-2">
          <button
            v-for="backend in RUNTIME_BACKENDS[currentRuntime]"
            :key="backend"
            :disabled="isBusy || isSwitchingBackend"
            :class="[
              'px-4 py-2 rounded font-medium transition-colors',
              currentBackend === backend
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
              (isBusy || isSwitchingBackend) && 'opacity-50 cursor-not-allowed'
            ]"
            @click="handleBackendChange(backend)"
          >
            {{ BACKEND_LABELS[backend] }}
          </button>
        </div>
        <p class="text-xs text-gray-500 mt-1">
          WebGPU: usually fastest (Chrome 113+) | WebGL: TensorFlow.js only | WASM: runs on the CPU
        </p>
      </div>

      <!-- Model variant selector -->
      <div class="mb-4">
        <label class="block text-sm text-gray-400 mb-2">Model Variant:</label>
        <div class="flex gap-2">
          <button
            v-for="variant in SUPPORTED_MODEL_VARIANTS"
            :key="variant"
            :disabled="isBusy || !isAvailable(variant, selectedSize)"
            :class="[
              'px-4 py-2 rounded font-medium transition-colors',
              currentModelVariant === variant
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
              (isBusy || !isAvailable(variant, selectedSize)) && 'opacity-50 cursor-not-allowed'
            ]"
            @click="handleVariantChange(variant)"
          >
            {{ variant === 's' ? 'S (standard)' : 'Tiny (lightweight)' }}
          </button>
        </div>
      </div>

      <!-- Model size selector -->
      <div class="mb-4">
        <label class="block text-sm text-gray-400 mb-2">Model Resolution:</label>
        <div class="flex gap-2">
          <button
            v-for="size in SUPPORTED_MODEL_SIZES"
            :key="size"
            :disabled="isBusy || !isAvailable(selectedVariant, size)"
            :class="[
              'px-4 py-2 rounded font-medium transition-colors',
              currentModelSize === size
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600',
              (isBusy || !isAvailable(selectedVariant, size)) && 'opacity-50 cursor-not-allowed'
            ]"
            @click="handleSizeChange(size)"
          >
            {{ size }}px
          </button>
        </div>
        <p class="text-xs text-gray-500 mt-1">
          Available models ({{ MODEL_VERSION }}): {{ AVAILABLE_MODELS.join(' / ') }} (input: BGR, stretched to S x S)
        </p>
      </div>

      <!-- Input mode selector -->
      <div class="mb-4 flex gap-4 items-center">
        <button
          v-if="!isCameraActive"
          :disabled="isLoading"
          class="px-4 py-2 rounded font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          @click="startCamera"
        >
          Start camera
        </button>
        <button
          v-else
          class="px-4 py-2 rounded font-medium bg-red-600 text-white hover:bg-red-700"
          @click="stopCamera"
        >
          Stop camera
        </button>

        <input
          type="file"
          accept="image/*"
          :disabled="isLoading || isCameraActive"
          class="block text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 disabled:opacity-50"
          @change="handleFileUpload"
        >
      </div>

      <!-- Inference info -->
      <div v-if="inferenceTime !== null" class="mb-4 text-gray-400">
        <span v-if="isCameraActive">
          {{ inferenceTime.toFixed(0) }}ms/frame | {{ fps }} FPS
        </span>
        <span v-else>
          Inference time: {{ inferenceTime.toFixed(0) }}ms ({{ result?.modelVariant }}_{{ result?.modelSize }})
        </span>
        <span v-if="result"> | Detected: {{ result.validCount }} person(s)</span>
      </div>

      <!-- Video container (camera mode) -->
      <div
        v-if="isCameraMode"
        style="position: relative; display: inline-block; max-height: 70vh;"
      >
        <video
          ref="videoRef"
          autoplay
          playsinline
          muted
          style="max-width: 100%; max-height: 70vh; display: block; transform: scaleX(-1);"
        />
        <canvas
          ref="canvasRef"
          style="position: absolute; top: 0; left: 0; pointer-events: none;"
        />
      </div>

      <!-- Image container with canvas overlay -->
      <div
        v-else-if="imageUrl"
        style="position: relative; display: inline-block; max-height: 70vh;"
      >
        <img
          ref="imageRef"
          :src="imageUrl"
          style="max-width: 100%; max-height: 70vh; display: block;"
          @load="handleImageLoad"
        >
        <canvas
          ref="canvasRef"
          style="position: absolute; top: 0; left: 0; pointer-events: none;"
        />
        <div
          v-if="isInferring"
          class="absolute inset-0 bg-black/50 flex items-center justify-center"
        >
          <span class="text-white text-lg">Processing...</span>
        </div>
      </div>

      <!-- Legends -->
      <div v-if="result && viewMode === 'parts'" class="mt-4 flex gap-2 text-xs">
        <span
          v-for="part in PART_COLORS"
          :key="part.ch"
          class="px-2 py-1 rounded text-black"
          :style="{ backgroundColor: `rgb(${part.rgb[0]}, ${part.rgb[1]}, ${part.rgb[2]})` }"
        >
          {{ part.name }}
        </span>
      </div>
      <div v-if="result && viewMode === 'depth'" class="mt-4 text-xs text-gray-400">
        Depth: red = near / blue = far (normalized per person around the pelvis = 0.5)
      </div>
      <div v-if="result && result.validCount > 0 && viewMode === 'kpts'" class="mt-4 text-sm text-gray-400">
        <div class="font-semibold mb-2">Head keypoints:</div>
        <div class="flex flex-wrap gap-2">
          <span
            v-for="kp in HEAD_KEYPOINTS"
            :key="kp.name"
            class="px-2 py-1 rounded text-xs"
            :style="{
              backgroundColor: `rgb(${kp.rgb[0]}, ${kp.rgb[1]}, ${kp.rgb[2]})`,
              color: textColorFor(kp.rgb)
            }"
          >
            {{ kp.name }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
