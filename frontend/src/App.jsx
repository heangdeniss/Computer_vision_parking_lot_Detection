import { useEffect, useRef, useState } from 'react'

const HEIC = /\.(heic|heif)$/i
const LABELS = { empty: 'Empty', occupied: 'Occupied' }

function Badges({ counts }) {
  return (
    <div className="counts">
      {Object.entries(counts).map(([name, count]) => (
        <span key={name} className={`badge ${name}`}>{LABELS[name] || name}: {count}</span>
      ))}
    </div>
  )
}

function drawLiveFrame(canvas, source, detections) {
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

export default function App() {
  const [tab, setTab] = useState('images')
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [video, setVideo] = useState(null)
  const [conf, setConf] = useState(0.25)
  const [stride, setStride] = useState(15)
  const [images, setImages] = useState(null)
  const [videoResult, setVideoResult] = useState(null)
  const [frame, setFrame] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [realtime, setRealtime] = useState(false)
  const [live, setLive] = useState(false)
  const [liveStats, setLiveStats] = useState(null)
  const dragDepth = useRef(0)
  const blobUrls = useRef(new Set())
  const inputRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const videoUrl = useRef(null)
  const confRef = useRef(conf)
  const liveRef = useRef({ active: false, busy: false, raf: 0, stream: null, last: 0, frames: 0, since: 0 })
  confRef.current = conf

  function releaseUrls() {
    blobUrls.current.forEach((url) => URL.revokeObjectURL(url))
    blobUrls.current.clear()
  }

  function stopLive() {
    const state = liveRef.current
    state.active = false
    cancelAnimationFrame(state.raf)
    const el = videoRef.current
    if (el) { el.pause(); el.srcObject = null; el.removeAttribute('src'); el.load() }
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop())
      state.stream = null
    }
    setLive(false)
    setLiveStats(null)
  }

  useEffect(() => () => {
    stopLive()
    releaseUrls()
    if (videoUrl.current) URL.revokeObjectURL(videoUrl.current)
  }, [])

  function switchTab(next) {
    if (live) stopLive()
    setTab(next)
    setError('')
    setLoading(false)
  }

  function clear() {
    if (live) stopLive()
    releaseUrls()
    setFiles([])
    setPreviews([])
    setVideo(null)
    setImages(null)
    setVideoResult(null)
    setFrame(0)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function addFiles(list) {
    const all = Array.from(list || [])
    setError('')
    if (tab === 'video') {
      setVideo(all.find((file) => file.type.startsWith('video/')) || null)
      setVideoResult(null)
      setFrame(0)
      return
    }
    // Drop can include folders/other files — keep only images, and append to the selection.
    const picked = all.filter((file) => file.type.startsWith('image/') || HEIC.test(file.name))
    if (!picked.length) {
      if (all.length) setError('Only image files can be added in the Images tab.')
      return
    }
    const urls = await Promise.all(picked.map(async (file) => {
      if (!HEIC.test(file.name)) {
        const url = URL.createObjectURL(file)
        blobUrls.current.add(url)
        return url
      }
      const body = new FormData()
      body.append('file', file)
      try {
        const response = await fetch('/api/preview', { method: 'POST', body })
        return (await response.json()).image || ''
      } catch {
        return ''
      }
    }))
    setFiles((current) => [...current, ...picked])
    setPreviews((current) => [...current, ...urls])
    setImages(null)
  }

  function removeFile(index) {
    const url = previews[index]
    if (url?.startsWith('blob:')) {
      URL.revokeObjectURL(url)
      blobUrls.current.delete(url)
    }
    setFiles((current) => current.filter((_file, i) => i !== index))
    setPreviews((current) => current.filter((_url, i) => i !== index))
    setImages(null)
  }

  function onDragEnter(event) {
    event.preventDefault()
    dragDepth.current += 1
    setDragging(true)
  }

  function onDragLeave(event) {
    event.preventDefault()
    dragDepth.current -= 1
    if (dragDepth.current <= 0) {
      dragDepth.current = 0
      setDragging(false)
    }
  }

  function onDrop(event) {
    event.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    addFiles(event.dataTransfer.files)
  }

  // One request at a time: if the API is still busy, skip this frame and keep playing.
  function tick() {
    const state = liveRef.current
    if (!state.active) return
    state.raf = requestAnimationFrame(tick)
    const el = videoRef.current
    const canvas = canvasRef.current
    if (!el || !canvas || el.readyState < 2 || state.busy) return
    const now = performance.now()
    if (now - state.last < 80) return
    state.last = now

    const scale = Math.min(1, 960 / el.videoWidth)
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
    state.busy = true
    shot.toBlob(async (blob) => {
      if (!blob) { state.busy = false; return }
      const body = new FormData()
      body.append('file', blob, 'frame.jpg')
      body.append('conf', String(confRef.current))
      try {
        const response = await fetch('/api/frame', { method: 'POST', body })
        if (response.ok && state.active) {
          const data = await response.json()
          drawLiveFrame(canvas, shot, data.detections)
          state.frames += 1
          if (now - state.since >= 1000) {
            setLiveStats({ counts: data.counts, fps: Math.round(state.frames * 1000 / (now - state.since)) })
            state.frames = 0
            state.since = now
          }
        }
      } catch {
        // Keep streaming if one frame fails.
      }
      state.busy = false
    }, 'image/jpeg', 0.7)
  }

  async function startLive() {
    setError('')
    const el = videoRef.current
    if (!el) return
    try {
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
        liveRef.current.stream = stream
        el.removeAttribute('src')
        el.srcObject = stream
      }
      el.muted = true
      el.playsInline = true
      await el.play()
      const state = liveRef.current
      state.active = true
      state.busy = false
      state.last = 0
      state.frames = 0
      state.since = performance.now()
      setLive(true)
      state.raf = requestAnimationFrame(tick)
    } catch (err) {
      setError(err.message || 'Could not start live video')
    }
  }

  function onSubmit(event) {
    event.preventDefault()
    if (tab === 'images') return detectImages(event)
    if (realtime) return live ? stopLive() : startLive()
    return detectVideo(event)
  }

  async function detectImages(event) {
    event.preventDefault()
    if (!files.length) return
    setLoading(true)
    setError('')
    const body = new FormData()
    files.forEach((file) => body.append('files', file))
    body.append('conf', String(conf))
    try {
      const response = await fetch('/api/predict', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'prediction failed')
      setImages(data.results)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function detectVideo(event) {
    event.preventDefault()
    if (!video) return
    setLoading(true)
    setError('')
    const body = new FormData()
    body.append('file', video)
    body.append('conf', String(conf))
    body.append('stride', String(stride))
    try {
      const response = await fetch('/api/video', { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'video processing failed')
      setVideoResult(data)
      setFrame(0)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const selected = videoResult?.results[frame]

  return (
    <main>
      <header>
        <h1>Parking occupancy detector</h1>
        <p>Upload images or a video to count empty and occupied spaces with YOLOv11.</p>
      </header>

      <nav className="tabs" aria-label="Input type">
        <button className={tab === 'images' ? 'active' : ''} onClick={() => switchTab('images')}>Images</button>
        <button className={tab === 'video' ? 'active' : ''} onClick={() => switchTab('video')}>Video</button>
      </nav>

      {tab === 'video' && <video ref={videoRef} className="hidden-video" muted playsInline />}

      <form onSubmit={onSubmit}>
        <div
          className={`dropzone ${dragging ? 'dragging' : ''}`}
          onDragEnter={onDragEnter}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            id="file-input"
            type="file"
            multiple={tab === 'images'}
            accept={tab === 'images' ? 'image/*,.heic,.heif' : 'video/*'}
            onChange={(e) => { addFiles(e.target.files); e.target.value = '' }}
          />
          <label htmlFor="file-input">
            <strong>{tab === 'images' ? 'Drop images here' : 'Drop a video here'}</strong>
            <span>{tab === 'images'
              ? (files.length ? `${files.length} selected — drop more or click to add` : 'or click to browse (JPG, PNG, HEIC, multiple)')
              : (video ? video.name : 'or click to browse (MP4, MOV, AVI)')}</span>
          </label>
        </div>

        <div className="controls">
          <label className="slider">
            Confidence <strong>{conf.toFixed(2)}</strong>
            <input type="range" min="0.05" max="0.9" step="0.05" value={conf}
              onChange={(e) => setConf(Number(e.target.value))} />
          </label>
          {tab === 'video' && (
            <label className="check">
              <input type="checkbox" checked={realtime} onChange={(e) => {
                if (live) stopLive()
                setRealtime(e.target.checked)
              }} />
              Realtime (live boxes)
            </label>
          )}
          {tab === 'video' && !realtime && (
            <label className="field">
              Sample every
              <select value={stride} onChange={(e) => setStride(Number(e.target.value))}>
                {[1, 5, 15, 30, 60].map((n) => <option key={n} value={n}>{n} frames</option>)}
              </select>
            </label>
          )}
          <button type="submit" disabled={loading || (tab === 'images' ? !files.length : (!realtime && !video))}>
            {loading ? 'Detecting…' : tab === 'video' && realtime ? (live ? 'Stop live' : 'Start live') : 'Detect'}
          </button>
          <button type="button" className="secondary" onClick={clear}>Clear</button>
        </div>
      </form>

      <p className="hint">
        {tab === 'images'
          ? 'Drop or add multiple images in one go — they are inferred in a single optimized batch. Use confidence to trade missed spaces against false alarms.'
          : realtime
            ? 'Live mode draws boxes on each frame as it plays. Uses the uploaded video if one is selected, otherwise the webcam (allow camera access).'
            : 'Every Nth frame is detected in batches; click a frame below to inspect it. Cap is 240 frames per video.'}
      </p>
      {error && <p className="error">{error}</p>}

      {tab === 'images' && files.length > 0 && (
        <div className="grid">
          {files.map((file, index) => {
            const result = images?.[index]
            const image = result?.image || previews[index]
            return (
              <section key={`${file.name}-${index}`}>
                <div className="card-head">
                  <h2>{file.name}</h2>
                  <button type="button" className="remove" onClick={() => removeFile(index)}
                    aria-label={`Remove ${file.name}`} title="Remove image">×</button>
                </div>
                {image ? <img src={image} alt={file.name} /> : <p className="placeholder">HEIC preview unavailable</p>}
                {result && (
                  <>
                    <Badges counts={result.counts} />
                    <details>
                      <summary>{result.detections.length} detections</summary>
                      <DetectionTable detections={result.detections} />
                    </details>
                  </>
                )}
              </section>
            )
          })}
        </div>
      )}

      {tab === 'video' && live && (
        <section className="video-result">
          <h2>Live detection — {video ? video.name : 'webcam'}</h2>
          <div className="live-stage">
            <canvas ref={canvasRef} />
          </div>
          <p className="meta">{liveStats ? `${liveStats.fps} fps · live inference` : 'Starting live stream…'}</p>
          {liveStats && <Badges counts={liveStats.counts} />}
        </section>
      )}

      {tab === 'video' && videoResult && (
        <section className="video-result">
          <h2>Frame timeline — {videoResult.name}</h2>
          <p className="meta">
            {videoResult.sampled} sampled of {videoResult.total_frames} frames · {videoResult.fps} fps · every {videoResult.stride} frames
          </p>
          <p className="meta">Totals across sampled frames:</p>
          <Badges counts={videoResult.totals} />

          <div className="timeline">
            {videoResult.results.map((item, index) => (
              <button type="button" key={item.index}
                className={`frame ${index === frame ? 'active' : ''}`}
                onClick={() => setFrame(index)}>
                <img src={item.image} alt={`frame ${item.index}`} />
                <span>#{item.index} · {item.time}s</span>
                <span>{item.counts.empty} empty / {item.counts.occupied} occupied</span>
              </button>
            ))}
          </div>

          {selected && (
            <div className="frame-detail">
              <div className="detail-head">
                <h3>Frame #{selected.index} at {selected.time}s</h3>
                <a href={selected.image} download={`frame_${selected.index}.jpg`}>Download annotated frame</a>
              </div>
              <img src={selected.image} alt={`annotated frame ${selected.index}`} />
              <Badges counts={selected.counts} />
              <DetectionTable detections={selected.detections} />
            </div>
          )}
        </section>
      )}
    </main>
  )
}

function DetectionTable({ detections }) {
  return (
    <table>
      <thead><tr><th>Class</th><th>Conf.</th><th>Box (x1, y1, x2, y2)</th></tr></thead>
      <tbody>
        {detections.map((d, i) => (
          <tr key={i}>
            <td><span className={`badge ${d.class}`}>{LABELS[d.class] || d.class}</span></td>
            <td>{d.confidence.toFixed(3)}</td>
            <td className="numbers">{d.bbox.join(', ')}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
