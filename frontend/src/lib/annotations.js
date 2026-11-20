import { LABELS } from './constants'

/** Draw the latest frame and its boxes on the live canvas. */
export function drawLiveFrame(canvas, source, detections) {
  const ctx = canvas.getContext('2d')
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height)
  ctx.font = 'bold 14px system-ui'
  for (const detection of detections) {
    const [x1, y1, x2, y2] = detection.bbox
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
