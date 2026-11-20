import { useCallback, useEffect, useRef, useState } from 'react'
import { postForm } from '../lib/api'
import { drawLiveFrame } from '../lib/annotations'

const MIN_INTERVAL_MS = 80
const MAX_WIDTH = 960

/** Streams frames from a video or webcam to /api/frame, one request at a time. */
export function useLiveDetection(conf, video) {
  const [live, setLive] = useState(false)
  const [stats, setStats] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const videoUrl = useRef(null)
  const confRef = useRef(conf)
  const state = useRef({ active: false, busy: false, raf: 0, stream: null, last: 0, frames: 0, since: 0 })
  confRef.current = conf

  const stop = useCallback(() => {
    const current = state.current
    current.active = false
    cancelAnimationFrame(current.raf)
    const el = videoRef.current
    if (el) { el.pause(); el.srcObject = null; el.removeAttribute('src'); el.load() }
    if (current.stream) {
      current.stream.getTracks().forEach((track) => track.stop())
      current.stream = null
    }
    setLive(false)
    setStats(null)
  }, [])

  const tick = useCallback(() => {
    const current = state.current
    if (!current.active) return
    current.raf = requestAnimationFrame(tick)
    const el = videoRef.current
    const canvas = canvasRef.current
    if (!el || !canvas || el.readyState < 2 || current.busy) return
    const now = performance.now()
    if (now - current.last < MIN_INTERVAL_MS) return
    current.last = now

    const scale = Math.min(1, MAX_WIDTH / el.videoWidth)
    const width = Math.round(el.videoWidth * scale)
    const height = Math.round(el.videoHeight * scale)
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }
    canvas.getContext('2d').drawImage(el, 0, 0, width, height)

    const shot = document.createElement('canvas')
    shot.width = width
    shot.height = height
    shot.getContext('2d').drawImage(el, 0, 0, width, height)
    current.busy = true
    shot.toBlob(async (blob) => {
      if (!blob) { current.busy = false; return }
      const body = new FormData()
      body.append('file', blob, 'frame.jpg')
      body.append('conf', String(confRef.current))
      try {
        const data = await postForm('/api/frame', body)
        if (current.active) {
          drawLiveFrame(canvas, shot, data.detections)
          current.frames += 1
          if (now - current.since >= 1000) {
            setStats({
              counts: data.counts,
              fps: Math.round(current.frames * 1000 / (now - current.since)),
            })
            current.frames = 0
            current.since = now
          }
        }
      } catch {
        // Keep streaming if a single frame fails.
      }
      current.busy = false
    }, 'image/jpeg', 0.7)
  }, [])

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
    Object.assign(state.current, {
      active: true, busy: false, last: 0, frames: 0, since: performance.now(),
    })
    setLive(true)
    state.current.raf = requestAnimationFrame(tick)
  }, [tick, video])

  useEffect(() => () => {
    stop()
    if (videoUrl.current) URL.revokeObjectURL(videoUrl.current)
  }, [stop])

  return { videoRef, canvasRef, live, stats, start, stop }
}
