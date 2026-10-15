// Draws an inference result (person masks, head / body keypoints, part masks, depth).
// The style follows tools/viewer.py of rtmdet-fusion.
import type { InferenceResult } from '~/composables/useRtmdetModel'

export const VIEW_MODES = ['kpts', 'parts', 'depth'] as const
export type ViewMode = (typeof VIEW_MODES)[number]
export const VIEW_LABELS: Record<ViewMode, string> = {
  kpts: 'Keypoints',
  parts: 'Part masks',
  depth: 'Depth',
}

// Head, 11 keypoints (same as SKELETON / KEYPOINT_COLORS in configs/default.py)
const HEAD_SKELETON: [number, number][] = [
  [1, 0], [0, 2], [0, 3], [4, 5], [6, 0], [7, 6], [0, 8], [1, 9], [9, 8], [0, 9], [0, 10],
]
export const HEAD_KEYPOINTS: { name: string; rgb: [number, number, number] }[] = [
  { name: 'center', rgb: [0, 255, 0] },
  { name: 'top', rgb: [0, 0, 255] },
  { name: 'left_ear', rgb: [255, 0, 255] },
  { name: 'right_ear', rgb: [200, 0, 200] },
  { name: 'left_eye', rgb: [0, 255, 255] },
  { name: 'right_eye', rgb: [0, 200, 200] },
  { name: 'nose', rgb: [255, 0, 0] },
  { name: 'chin', rgb: [255, 128, 0] },
  { name: 'occiput', rgb: [128, 128, 128] },
  { name: 'vertex', rgb: [255, 255, 0] },
  { name: 'neck', rgb: [0, 200, 255] },
]
const HEAD_COLORS = HEAD_KEYPOINTS.map(({ rgb }) => `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`)

// Whole body, 31 keypoints (same as BODY_SKELETON / BODY_KEYPOINT_COLORS_RGB in configs/default.py)
const BODY_SKELETON: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [11, 12], [5, 11], [6, 12],
  [11, 13], [13, 15], [12, 14], [14, 16], [17, 11], [17, 12], [17, 18], [18, 19], [19, 20], [0, 20],
  [19, 21], [19, 22], [21, 5], [22, 6], [15, 23], [15, 25], [16, 24], [16, 26], [9, 27], [9, 29],
  [10, 28], [10, 30],
]
const BODY_COLORS = [
  [255, 0, 0], [255, 165, 0], [80, 255, 80], [255, 165, 0], [80, 255, 80], [255, 165, 0], [80, 255, 80],
  [255, 165, 0], [80, 255, 80], [255, 165, 0], [80, 255, 80], [255, 165, 0], [80, 255, 80], [255, 165, 0],
  [80, 255, 80], [255, 165, 0], [80, 255, 80], [255, 255, 255], [0, 255, 255], [0, 255, 255], [255, 0, 0],
  [255, 165, 0], [80, 255, 80], [255, 165, 0], [80, 255, 80], [255, 165, 0], [80, 255, 80], [255, 100, 0],
  [0, 200, 0], [255, 100, 0], [0, 200, 0],
].map(c => `rgb(${c[0]},${c[1]},${c[2]})`)

// Part mask channels: 0=clothed body 1=bare body 2=head 3=torso 4=limbs 5=hands 6=occluded
const NUM_PART_CHANNELS = 7
export const PART_COLORS: { ch: number; name: string; rgb: [number, number, number] }[] = [
  { ch: 2, name: 'Head', rgb: [255, 0, 0] },
  { ch: 3, name: 'Torso', rgb: [0, 200, 0] },
  { ch: 4, name: 'Limbs', rgb: [0, 128, 255] },
  { ch: 5, name: 'Hands', rgb: [255, 255, 0] },
]

const PERSON_COLORS: [number, number, number][] = [
  [100, 100, 255], [100, 255, 100], [255, 100, 100], [100, 255, 255], [255, 100, 255],
]

// Turbo colormap (Google's polynomial approximation). t: 0=blue ... 1=red
function turbo(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t))
  const r = 34.61 + x * (1172.33 - x * (10793.56 - x * (33300.12 - x * (38394.49 - x * 14825.05))))
  const g = 23.31 + x * (557.33 + x * (1225.33 - x * (3574.96 - x * (1073.77 + x * 707.56))))
  const b = 27.2 + x * (3211.1 - x * (15327.97 - x * (27814 - x * (22569.18 - x * 6838.66))))
  const c = (v: number) => Math.max(0, Math.min(255, Math.round(v)))
  return [c(r), c(g), c(b)]
}

let offscreen: HTMLCanvasElement | null = null
function getOffscreen(size: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  if (!offscreen) offscreen = document.createElement('canvas')
  if (offscreen.width !== size) {
    offscreen.width = size
    offscreen.height = size
  }
  return { canvas: offscreen, ctx: offscreen.getContext('2d')! }
}

/** Relative keypoint -> model coordinates using its reference bbox (rel_z is 1 at the top) */
function kptToModel(kp: number[], bbox: number[]): [number, number] {
  const [x1, y1, x2, y2] = bbox as [number, number, number, number]
  return [x1 + (kp[0] ?? 0.5) * (x2 - x1), y1 + (1 - (kp[2] ?? 0.5)) * (y2 - y1)]
}

/** Stretches a size x size RGBA buffer over the whole display canvas */
function blitRgba(ctx: CanvasRenderingContext2D, rgba: Uint8ClampedArray<ArrayBuffer>, size: number, w: number, h: number) {
  const { canvas, ctx: oc } = getOffscreen(size)
  oc.putImageData(new ImageData(rgba, size, size), 0, 0)
  ctx.imageSmoothingEnabled = true
  ctx.drawImage(canvas, 0, 0, w, h)
}

/**
 * Draws an inference result. With mirror=true (camera preview) everything is flipped horizontally.
 * Coordinates are in the model's S x S space (the input was stretched without keeping the aspect ratio).
 */
export function drawInferenceResult(
  ctx: CanvasRenderingContext2D,
  result: InferenceResult,
  displayWidth: number,
  displayHeight: number,
  mode: ViewMode,
  mirror = false,
) {
  const modelSize = result.modelSize
  const sx = displayWidth / modelSize
  const sy = displayHeight / modelSize
  const n = Math.min(result.validCount, result.scores.length)
  const pm = result.partMaskSize
  const pmArea = pm * pm

  // Index of part_masks[i, ch, p] (p = pixel index) for either memory layout
  const channelsLast = result.partMaskLayout === 'channels-last'
  const instStride = pmArea * NUM_PART_CHANNELS
  const pixStride = channelsLast ? NUM_PART_CHANNELS : 1
  const chStride = channelsLast ? 1 : pmArea
  const partMasks = result.partMasks

  ctx.save()
  if (mirror) {
    ctx.translate(displayWidth, 0)
    ctx.scale(-1, 1)
  }

  if (mode === 'parts' || mode === 'depth') {
    const rgba = new Uint8ClampedArray(pmArea * 4)
    for (let i = 0; i < n; i++) {
      const base = i * instStride
      if (mode === 'parts') {
        for (let p = 0; p < pmArea; p++) {
          const off = base + p * pixStride
          for (const { ch, rgb } of PART_COLORS) {
            if ((partMasks[off + ch * chStride] ?? 0) > 0.5) {
              rgba[p * 4] = rgb[0]
              rgba[p * 4 + 1] = rgb[1]
              rgba[p * 4 + 2] = rgb[2]
              rgba[p * 4 + 3] = 150
            }
          }
        }
      } else {
        // Depth inside the clothed-body mask (ch0), normalized over a range symmetric around
        // the pelvis (0.5): near = red / far = blue
        const depthBase = i * pmArea
        let radius = 1e-3
        for (let p = 0; p < pmArea; p++) {
          if ((partMasks[base + p * pixStride] ?? 0) > 0.5) {
            radius = Math.max(radius, Math.abs((result.depth[depthBase + p] ?? 0.5) - 0.5))
          }
        }
        for (let p = 0; p < pmArea; p++) {
          if ((partMasks[base + p * pixStride] ?? 0) <= 0.5) continue
          const t = ((result.depth[depthBase + p] ?? 0.5) - 0.5 + radius) / (2 * radius)
          const [r, g, b] = turbo(1 - t)
          rgba[p * 4] = r
          rgba[p * 4 + 1] = g
          rgba[p * 4 + 2] = b
          rgba[p * 4 + 3] = 190
        }
      }
    }
    blitRgba(ctx, rgba, pm, displayWidth, displayHeight)
  }

  if (mode === 'kpts') {
    // Person instance masks
    const area = modelSize * modelSize
    const rgba = new Uint8ClampedArray(area * 4)
    for (let i = 0; i < n; i++) {
      const [r, g, b] = PERSON_COLORS[i % PERSON_COLORS.length]!
      for (let p = 0; p < area; p++) {
        if ((result.masks[i * area + p] ?? 0) > 0.5) {
          rgba[p * 4] = r
          rgba[p * 4 + 1] = g
          rgba[p * 4 + 2] = b
          rgba[p * 4 + 3] = 110
        }
      }
    }
    blitRgba(ctx, rgba, modelSize, displayWidth, displayHeight)
  }

  for (let i = 0; i < n; i++) {
    const bbox = result.bboxes[i]!
    ctx.strokeStyle = 'rgb(0,255,0)'
    ctx.lineWidth = 2
    ctx.strokeRect(bbox[0]! * sx, bbox[1]! * sy, (bbox[2]! - bbox[0]!) * sx, (bbox[3]! - bbox[1]!) * sy)
    if (mode !== 'kpts') continue

    // Whole body (skeleton, then points; points predicted to be out of frame are skipped)
    const bodyBbox = result.bodyBboxes[i]!
    const bodyPts = result.bodyKeypoints[i]!.map(kp => kptToModel(kp, bodyBbox))
    const bodyIn = result.bodyInFrame[i]!
    ctx.strokeStyle = 'rgba(220,220,220,0.85)'
    ctx.lineWidth = 2
    for (const [a, b] of BODY_SKELETON) {
      if ((bodyIn[a] ?? 0) < 0.5 || (bodyIn[b] ?? 0) < 0.5) continue
      ctx.beginPath()
      ctx.moveTo(bodyPts[a]![0] * sx, bodyPts[a]![1] * sy)
      ctx.lineTo(bodyPts[b]![0] * sx, bodyPts[b]![1] * sy)
      ctx.stroke()
    }
    bodyPts.forEach(([x, y], k) => {
      if ((bodyIn[k] ?? 0) < 0.5) return
      ctx.beginPath()
      ctx.arc(x * sx, y * sy, 4, 0, Math.PI * 2)
      ctx.fillStyle = BODY_COLORS[k]!
      ctx.fill()
    })

    // Head (bbox, skeleton, points)
    const headBbox = result.headBboxes[i]!
    ctx.strokeStyle = 'rgba(0,255,0,0.8)'
    ctx.lineWidth = 1
    ctx.strokeRect(headBbox[0]! * sx, headBbox[1]! * sy, (headBbox[2]! - headBbox[0]!) * sx, (headBbox[3]! - headBbox[1]!) * sy)
    const headPts = result.keypoints[i]!.map(kp => kptToModel(kp, headBbox))
    ctx.strokeStyle = 'rgba(180,180,180,0.9)'
    for (const [a, b] of HEAD_SKELETON) {
      ctx.beginPath()
      ctx.moveTo(headPts[a]![0] * sx, headPts[a]![1] * sy)
      ctx.lineTo(headPts[b]![0] * sx, headPts[b]![1] * sy)
      ctx.stroke()
    }
    headPts.forEach(([x, y], k) => {
      ctx.beginPath()
      ctx.arc(x * sx, y * sy, 3.5, 0, Math.PI * 2)
      ctx.fillStyle = HEAD_COLORS[k]!
      ctx.fill()
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 1
      ctx.stroke()
    })
  }
  ctx.restore()
}
