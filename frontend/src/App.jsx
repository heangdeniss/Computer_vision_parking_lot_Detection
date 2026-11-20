import { useEffect, useRef, useState } from 'react'
import Controls from './components/Controls'
import ImageCard from './components/ImageCard'
import LiveView from './components/LiveView'
import UploadZone from './components/UploadZone'
import VideoTimeline from './components/VideoTimeline'
import { useLiveDetection } from './hooks/useLiveDetection'
import { postForm } from './lib/api'
import { HEIC } from './lib/constants'

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
  const dragDepth = useRef(0)
  const blobUrls = useRef(new Set())
  const inputRef = useRef(null)
  const { videoRef, canvasRef, live, stats: liveStats, start, stop } = useLiveDetection(conf, video)

  function releaseUrls() {
    blobUrls.current.forEach((url) => URL.revokeObjectURL(url))
    blobUrls.current.clear()
  }

  useEffect(() => () => releaseUrls(), [])

  function switchTab(next) {
    if (live) stop()
    setTab(next)
    setError('')
    setLoading(false)
  }

  function clear() {
    if (live) stop()
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
        return (await postForm('/api/preview', body)).image || ''
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

  function onDragOver(event) {
    event.preventDefault()
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

  async function detectImages() {
    if (!files.length) return
    setLoading(true)
    setError('')
    const body = new FormData()
    files.forEach((file) => body.append('files', file))
    body.append('conf', String(conf))
    try {
      setImages((await postForm('/api/predict', body)).results)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function detectVideo() {
    if (!video) return
    setLoading(true)
    setError('')
    const body = new FormData()
    body.append('file', video)
    body.append('conf', String(conf))
    body.append('stride', String(stride))
    try {
      setVideoResult(await postForm('/api/video', body))
      setFrame(0)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function toggleLive() {
    setError('')
    if (live) {
      stop()
      return
    }
    try {
      await start()
    } catch (err) {
      setError(err.message || 'Could not start live video')
    }
  }

  function onSubmit(event) {
    event.preventDefault()
    if (tab === 'images') return detectImages()
    if (realtime) return toggleLive()
    return detectVideo()
  }

  const hint = tab === 'images'
    ? 'Drop or add multiple images in one go — they are inferred in a single optimized batch. Use confidence to trade missed spaces against false alarms.'
    : realtime
      ? 'Live mode draws boxes on each frame as it plays. Uses the uploaded video if one is selected, otherwise the webcam (allow camera access).'
      : 'Every Nth frame is detected in batches; click a frame below to inspect it. Cap is 240 frames per video.'

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
        <UploadZone
          tab={tab}
          fileCount={files.length}
          videoName={video?.name}
          inputRef={inputRef}
          dragging={dragging}
          onFiles={addFiles}
          onDragEnter={onDragEnter}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        />
        <Controls
          tab={tab}
          conf={conf}
          stride={stride}
          realtime={realtime}
          live={live}
          loading={loading}
          canDetect={tab === 'images' ? files.length > 0 : (realtime || Boolean(video))}
          onConf={setConf}
          onStride={setStride}
          onRealtime={(value) => { if (live) stop(); setRealtime(value) }}
          onClear={clear}
        />
      </form>

      <p className="hint">{hint}</p>
      {error && <p className="error">{error}</p>}

      {tab === 'images' && files.length > 0 && (
        <div className="grid">
          {files.map((file, index) => (
            <ImageCard key={`${file.name}-${index}`}
              file={file}
              preview={previews[index]}
              result={images?.[index]}
              onRemove={() => removeFile(index)} />
          ))}
        </div>
      )}

      {tab === 'video' && live && (
        <LiveView name={video ? video.name : 'webcam'} canvasRef={canvasRef} stats={liveStats} />
      )}

      {tab === 'video' && videoResult && (
        <VideoTimeline result={videoResult} frame={frame} onSelect={setFrame} />
      )}
    </main>
  )
}
