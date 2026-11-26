import { useCallback, useEffect, useRef, useState } from 'react'
import { postForm } from '../lib/api'
import { drawLiveFrame } from '../lib/annotations'

const DISPLAY_MAX_WIDTH = 960
const INFER_MAX_WIDTH = 640
const STATS_INTERVAL_MS = 500

/**
 * Real-time detection loop decoupled from display rendering:
 * - requestAnimationFrame continuously draws the live video at native 30/60fps
 *   and overlays the latest boxes without flicker or frame rewind.
 * - Each active model runs its own back-to-back async inference loop at 640px
 *   so YOLOv11 updates in true real time (~30fps) without waiting on Faster R-CNN.
 */
export function useLiveDetection(conf, video, model = 'yolo11') {
  const [live, setLive] = useState(false)
  const [stats, setStats] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const compareCanvasRef = useRef(null)
  const videoUrl = useRef(null)
  const confRef = useRef(conf)
  const modelRef = useRef(model)
  const state = useRef({
    active: false,
    runId: 0,
    raf: 0,
    stream: null,
    latest: {
      yolo11: { detections: [], scaleX: 1, scaleY: 1, counts: null, latency_ms: null, label: 'YOLOv11s', fps: 0 },
      faster_rcnn: { detections: [], scaleX: 1, scaleY: 1, counts: null, latency_ms: null, label: 'Faster R-CNN', fps: 0 },
    },
  })
  confRef.current = conf
  modelRef.current = model

  const stop = useCallback(() => {
    const current = state.current
    current.active = false
    current.runId += 1
    cancelAnimationFrame(current.raf)
    const el = videoRef.current
    if (el) { el.pause(); el.srcObject = null; el.removeAttribute('src'); el.load() }
    if (current.stream) {
      current.stream.getTracks().forEach((track) => track.stop())
      current.stream = null
    }
    current.latest.yolo11 = { detections: [], scaleX: 1, scaleY: 1, counts: null, latency_ms: null, label: 'YOLOv11s', fps: 0 }
    current.latest.faster_rcnn = { detections: [], scaleX: 1, scaleY: 1, counts: null, latency_ms: null, label: 'Faster R-CNN', fps: 0 }
    setLive(false)
    setStats(null)
  }, [])

  const publishStats = useCallback(() => {
    const current = state.current
    if (!current.active) return
    const activeModel = modelRef.current
    if (activeModel === 'both') {
      const y = current.latest.yolo11
      const f = current.latest.faster_rcnn
      setStats({
        mode: 'both',
        fps: y.fps,
        models: {
          yolo11: y.counts ? { counts: y.counts, latency_ms: y.latency_ms, label: y.label, fps: y.fps } : null,
          faster_rcnn: f.counts ? { counts: f.counts, latency_ms: f.latency_ms, label: f.label, fps: f.fps } : null,
        },
      })
    } else {
      const entry = current.latest[activeModel]
      if (entry?.counts) {
        setStats({
          counts: entry.counts,
          latency_ms: entry.latency_ms,
          label: entry.label,
          fps: entry.fps,
        })
      }
    }
  }, [])

  const renderLoop = useCallback(() => {
    const current = state.current
    if (!current.active) return
    current.raf = requestAnimationFrame(renderLoop)
    const el = videoRef.current
    const canvas = canvasRef.current
    const compareCanvas = compareCanvasRef.current
    if (!el || !canvas || el.readyState < 2 || !el.videoWidth) return

    const scale = Math.min(1, DISPLAY_MAX_WIDTH / el.videoWidth)
    const width = Math.round(el.videoWidth * scale)
    const height = Math.round(el.videoHeight * scale)
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

    const activeModel = modelRef.current
    if (activeModel === 'both') {
      const y = current.latest.yolo11
      drawLiveFrame(canvas, el, y.detections, y.scaleX, y.scaleY)
      if (compareCanvas) {
        if (compareCanvas.width !== width || compareCanvas.height !== height) {
          compareCanvas.width = width
          compareCanvas.height = height
        }
        const f = current.latest.faster_rcnn
        drawLiveFrame(compareCanvas, el, f.detections, f.scaleX, f.scaleY)
      }
    } else {
      const entry = current.latest[activeModel] || current.latest.yolo11
      drawLiveFrame(canvas, el, entry.detections, entry.scaleX, entry.scaleY)
    }
  }, [])

  const runModelWorker = useCallback(async (modelKey, runId) => {
    const current = state.current
    const shot = document.createElement('canvas')
    const shotCtx = shot.getContext('2d')
    let frames = 0
    let since = performance.now()

    while (current.active && current.runId === runId) {
      const el = videoRef.current
      if (!el || el.readyState < 2 || !el.videoWidth) {
        await new Promise((r) => setTimeout(r, 16))
        continue
      }

      const displayScale = Math.min(1, DISPLAY_MAX_WIDTH / el.videoWidth)
      const displayW = Math.round(el.videoWidth * displayScale)
      const displayH = Math.round(el.videoHeight * displayScale)

      const inferScale = Math.min(1, INFER_MAX_WIDTH / el.videoWidth)
      const inferW = Math.round(el.videoWidth * inferScale)
      const inferH = Math.round(el.videoHeight * inferScale)

      if (shot.width !== inferW || shot.height !== inferH) {
        shot.width = inferW
        shot.height = inferH
      }
      shotCtx.drawImage(el, 0, 0, inferW, inferH)

      const blob = await new Promise((resolve) => shot.toBlob(resolve, 'image/jpeg', 0.65))
      if (!blob || !current.active || current.runId !== runId) continue

      const body = new FormData()
      body.append('file', blob, 'frame.jpg')
      body.append('conf', String(confRef.current))
      body.append('model', modelKey)

      try {
        const data = await postForm('/api/frame', body)
        if (!current.active || current.runId !== runId) break
        const entry = current.latest[modelKey]
        entry.detections = data.detections
        entry.scaleX = displayW / inferW
        entry.scaleY = displayH / inferH
        entry.counts = data.counts
        entry.latency_ms = data.latency_ms
        if (data.label) entry.label = data.label

        frames += 1
        const now = performance.now()
        if (now - since >= STATS_INTERVAL_MS) {
          entry.fps = Math.round((frames * 1000) / (now - since))
          frames = 0
          since = now
          publishStats()
        } else if (entry.fps === 0) {
          publishStats()
        }
      } catch {
        await new Promise((r) => setTimeout(r, 100))
      }
    }
  }, [publishStats])

  const start = useCallback(async () => {
    const el = videoRef.current
    if (!el) return
    if (video) {
      if (videoUrl.current) URL.revokeObjectURL(videoUrl.current)
      videoUrl.current = URL.createObjectURL(video)
      el.srcObject = null
      el.src = videoUrl.current
      el.loop = true
    } else {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      })
      state.current.stream = stream
      el.removeAttribute('src')
      el.srcObject = stream
    }
    el.muted = true
    el.playsInline = true
    await el.play()

    const current = state.current
    current.runId += 1
    const runId = current.runId
    current.active = true
    setLive(true)
    current.raf = requestAnimationFrame(renderLoop)

    const activeModel = modelRef.current
    if (activeModel === 'both') {
      runModelWorker('yolo11', runId)
      runModelWorker('faster_rcnn', runId)
    } else {
      runModelWorker(activeModel, runId)
    }
  }, [renderLoop, runModelWorker, video])

  useEffect(() => () => {
    stop()
    if (videoUrl.current) URL.revokeObjectURL(videoUrl.current)
  }, [stop])

  return { videoRef, canvasRef, compareCanvasRef, live, stats, start, stop }
}
