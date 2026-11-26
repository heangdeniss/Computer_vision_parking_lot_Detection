import Badges from './Badges'

export default function LiveView({ name, model, canvasRef, compareCanvasRef, stats }) {
  const isCompare = model === 'both'

  if (isCompare) {
    const yolo = stats?.models?.yolo11
    const frcnn = stats?.models?.faster_rcnn
    return (
      <section className="video-result">
        <h2>Live side-by-side detection — {name}</h2>
        <p className="meta">
          {yolo || frcnn
            ? 'Independent real-time streams — each model updates at its own GPU frame rate'
            : 'Starting dual-model live stream…'}
        </p>
        <div className="compare-grid">
          <div className="compare-pane">
            <div className="pane-head">
              <span className="model-tag yolo">{yolo?.label || 'YOLOv11s'}</span>
              {yolo && (
                <span className="latency-pill">
                  {yolo.fps ? `${yolo.fps} fps · ` : ''}{yolo.latency_ms?.toFixed(1)} ms
                </span>
              )}
            </div>
            <div className="live-stage"><canvas ref={canvasRef} /></div>
            {yolo?.counts && <Badges counts={yolo.counts} />}
          </div>
          <div className="compare-pane">
            <div className="pane-head">
              <span className="model-tag frcnn">{frcnn?.label || 'Faster R-CNN'}</span>
              {frcnn && (
                <span className="latency-pill">
                  {frcnn.fps ? `${frcnn.fps} fps · ` : ''}{frcnn.latency_ms?.toFixed(1)} ms
                </span>
              )}
            </div>
            <div className="live-stage"><canvas ref={compareCanvasRef} /></div>
            {frcnn?.counts && <Badges counts={frcnn.counts} />}
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="video-result">
      <h2>Live detection — {name}</h2>
      <div className="live-stage"><canvas ref={canvasRef} /></div>
      <p className="meta">
        {stats
          ? `${stats.fps} fps · ${stats.label || model}${typeof stats.latency_ms === 'number' ? ` · ${stats.latency_ms.toFixed(1)} ms` : ''}`
          : 'Starting live stream…'}
      </p>
      {stats && <Badges counts={stats.counts} />}
    </section>
  )
}
