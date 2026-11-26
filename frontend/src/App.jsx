import { useEffect, useRef, useState } from 'react'
import Controls from './components/Controls'
import ImageCard from './components/ImageCard'
import LiveView from './components/LiveView'
import UploadZone from './components/UploadZone'
import VideoTimeline from './components/VideoTimeline'
import { useLiveDetection } from './hooks/useLiveDetection'
import { getJson, postForm } from './lib/api'
import { HEIC } from './lib/constants'

export default function App() {
  const [tab, setTab] = useState('images')
  const [model, setModel] = useState('both')
  const [hasFasterRcnn, setHasFasterRcnn] = useState(true)
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
  const {
    videoRef, canvasRef, compareCanvasRef, live, stats: liveStats, start, stop,
  } = useLiveDetection(conf, video, model)

  function releaseUrls() {
    blobUrls.current.forEach((url) => URL.revokeObjectURL(url))
    blobUrls.current.clear()
  }

  useEffect(() => () => releaseUrls(), [])

  useEffect(() => {
    getJson('/api/health')
      .then((info) => {
        const available = Boolean(info?.models?.faster_rcnn?.available)
        setHasFasterRcnn(available)
        if (!available) setModel('yolo11')
      })
      .catch(() => {})
  }, [])

  function switchTab(next) {
    if (live) stop()
    setTab(next)
    setError('')
    setLoading(false)
  }

  function changeModel(next) {
    if (live) stop()
    setModel(next)
    setImages(null)
    setVideoResult(null)
    setError('')
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
    body.append('model', model)
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
    body.append('model', model)
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
    ? model === 'both'
      ? 'Side-by-side mode runs YOLOv11 and Faster R-CNN on the same uploaded images so you can compare counts, boxes, and latency directly.'
      : 'Drop or add multiple images in one go — they are inferred in a single optimized batch. Use confidence to trade missed spaces against false alarms.'
    : realtime
      ? 'Live mode draws boxes on each frame as it plays (supports single model or dual side-by-side canvases). Uses the uploaded video if selected, otherwise the webcam.'
      : 'Every Nth frame is detected in batches; click a frame below to inspect it. Cap is 240 frames per video.'

  const showingCompareCards = Boolean(images?.length && images[0]?.models)

  return (
    <main>
      <header>
        <h1>Parking occupancy detector</h1>
        <p>Compare YOLOv11 and Faster R-CNN side by side on images, video timelines, and live camera streams.</p>
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
          model={model}
          hasFasterRcnn={hasFasterRcnn}
          conf={conf}
          stride={stride}
          realtime={realtime}
          live={live}
          loading={loading}
          canDetect={tab === 'images' ? files.length > 0 : (realtime || Boolean(video))}
          onModel={changeModel}
          onConf={setConf}
          onStride={setStride}
          onRealtime={(value) => { if (live) stop(); setRealtime(value) }}
          onClear={clear}
        />
      </form>

      <p className="hint">{hint}</p>
      {error && <p className="error">{error}</p>}

      {tab === 'images' && files.length > 0 && (
        <div className={showingCompareCards ? 'grid compare-list' : 'grid'}>
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
        <LiveView
          name={video ? video.name : 'webcam'}
          model={model}
          canvasRef={canvasRef}
          compareCanvasRef={compareCanvasRef}
          stats={liveStats}
        />
      )}

      {tab === 'video' && videoResult && (
        <VideoTimeline result={videoResult} frame={frame} onSelect={setFrame} />
      )}
    </main>
  )
}
