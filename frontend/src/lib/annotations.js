import { LABELS } from './constants'

/** Draw detection boxes onto a canvas already displaying the live video frame. */
export function drawLiveBoxes(ctx, detections, scaleX = 1, scaleY = 1) {
  if (!detections?.length) return
  ctx.font = 'bold 14px system-ui'
  for (const detection of detections) {
    const [rx1, ry1, rx2, ry2] = detection.bbox
    const x1 = rx1 * scaleX
    const y1 = ry1 * scaleY
    const x2 = rx2 * scaleX
    const y2 = ry2 * scaleY
    const color = detection.class === 'empty' ? '#00ff00' : '#ff0000'
    const label = `${LABELS[detection.class] || detection.class} ${detection.confidence.toFixed(2)}`
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    ctx.lineWidth = 4
    ctx.strokeStyle = '#000'
    ctx.strokeText(label, x1, Math.max(16, y1 - 6))
    ctx.fillStyle = color
    ctx.fillText(label, x1, Math.max(16, y1 - 6))
  }
}

/** Draw the live video frame and overlay the latest boxes with coordinate scaling. */
export function drawLiveFrame(canvas, source, detections, scaleX = 1, scaleY = 1) {
  const ctx = canvas.getContext('2d')
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height)
  drawLiveBoxes(ctx, detections, scaleX, scaleY)
}
