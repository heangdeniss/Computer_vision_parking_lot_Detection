import Badges from './Badges'
import DetectionTable from './DetectionTable'

export default function VideoTimeline({ result, frame, onSelect }) {
  const selected = result.results[frame]
  const isCompare = result.mode === 'both'

  if (isCompare) {
    const yoloMeta = result.models.yolo11
    const frcnnMeta = result.models.faster_rcnn
    return (
      <section className="video-result">
        <h2>Frame timeline (Side-by-side comparison) — {result.name}</h2>
        <p className="meta">
          {result.sampled} sampled of {result.total_frames} frames · {result.fps} fps · every {result.stride} frames
        </p>

        <div className="totals-compare">
          <div className="total-box">
            <div className="pane-head">
              <span className="model-tag yolo">{yoloMeta.label}</span>
              <span className="latency-pill">{yoloMeta.avg_latency_ms.toFixed(1)} ms/frame</span>
            </div>
            <Badges counts={yoloMeta.totals} />
          </div>
          <div className="total-box">
            <div className="pane-head">
              <span className="model-tag frcnn">{frcnnMeta.label}</span>
              <span className="latency-pill">{frcnnMeta.avg_latency_ms.toFixed(1)} ms/frame</span>
            </div>
            <Badges counts={frcnnMeta.totals} />
          </div>
        </div>

        <div className="timeline">
          {result.results.map((item, index) => {
            const y = item.models.yolo11
            const f = item.models.faster_rcnn
            return (
              <button type="button" key={item.index}
                className={`frame ${index === frame ? 'active' : ''}`}
                onClick={() => onSelect(index)}>
                <img src={y.image} alt={`frame ${item.index}`} />
                <span>#{item.index} · {item.time}s</span>
                <span>YOLO: {y.counts.empty}E / {y.counts.occupied}O</span>
                <span>FRCNN: {f.counts.empty}E / {f.counts.occupied}O</span>
              </button>
            )
          })}
        </div>

        {selected && (
          <div className="frame-detail">
            <div className="detail-head">
              <h3>Frame #{selected.index} at {selected.time}s</h3>
            </div>
            <div className="compare-grid">
              {[
                { key: 'yolo11', variant: 'yolo', data: selected.models.yolo11 },
                { key: 'faster_rcnn', variant: 'frcnn', data: selected.models.faster_rcnn },
              ].map(({ key, variant, data }) => (
                <div key={key} className="compare-pane">
                  <div className="pane-head">
                    <span className={`model-tag ${variant}`}>{data.label}</span>
                    <a href={data.image} download={`frame_${selected.index}_${key}.jpg`}>Download</a>
                  </div>
                  <img src={data.image} alt={`${data.label} frame ${selected.index}`} />
                  <Badges counts={data.counts} />
                  <details>
                    <summary>{data.detections.length} detections</summary>
                    <DetectionTable detections={data.detections} />
                  </details>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    )
  }

  return (
    <section className="video-result">
      <h2>Frame timeline — {result.name} {result.label ? `(${result.label})` : ''}</h2>
      <p className="meta">
        {result.sampled} sampled of {result.total_frames} frames · {result.fps} fps · every {result.stride} frames
        {typeof result.avg_latency_ms === 'number' ? ` · ${result.avg_latency_ms.toFixed(1)} ms/frame` : ''}
      </p>
      <p className="meta">Totals across sampled frames:</p>
      <Badges counts={result.totals} />

      <div className="timeline">
        {result.results.map((item, index) => (
          <button type="button" key={item.index}
            className={`frame ${index === frame ? 'active' : ''}`}
            onClick={() => onSelect(index)}>
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
  )
}
