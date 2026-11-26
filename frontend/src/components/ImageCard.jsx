import Badges from './Badges'
import DetectionTable from './DetectionTable'

function formatDelta(diff) {
  return diff > 0 ? `+${diff}` : String(diff)
}

function ModelPane({ tag, variant, data, fileName }) {
  return (
    <div className="compare-pane">
      <div className="pane-head">
        <span className={`model-tag ${variant}`}>{data.label || tag}</span>
        {typeof data.latency_ms === 'number' && (
          <span className="latency-pill">{data.latency_ms.toFixed(1)} ms</span>
        )}
      </div>
      <img src={data.image} alt={`${fileName} — ${data.label || tag}`} />
      <Badges counts={data.counts} />
      <details>
        <summary>{data.detections.length} detections</summary>
        <DetectionTable detections={data.detections} />
      </details>
    </div>
  )
}

export default function ImageCard({ file, preview, result, onRemove }) {
  const isCompare = Boolean(result?.models)

  if (isCompare) {
    const yolo = result.models.yolo11
    const frcnn = result.models.faster_rcnn
    const dEmpty = (frcnn?.counts?.empty ?? 0) - (yolo?.counts?.empty ?? 0)
    const dOcc = (frcnn?.counts?.occupied ?? 0) - (yolo?.counts?.occupied ?? 0)
    const yMs = yolo?.latency_ms || 0
    const fMs = frcnn?.latency_ms || 0
    const ratio = yMs > 0 && fMs > 0 ? `${(fMs / yMs).toFixed(1)}×` : null

    return (
      <section className="card-compare">
        <div className="card-head">
          <h2>{file.name}</h2>
          <button type="button" className="remove" onClick={onRemove}
            aria-label={`Remove ${file.name}`} title="Remove image">×</button>
        </div>
        <div className="compare-diff">
          <span>Faster R-CNN vs YOLOv11:</span>
          <span>Δ Empty <strong>{formatDelta(dEmpty)}</strong></span>
          <span>Δ Occupied <strong>{formatDelta(dOcc)}</strong></span>
          {ratio && <span>Speed: YOLOv11 <strong>{ratio}</strong> faster</span>}
        </div>
        <div className="compare-grid">
          {yolo && <ModelPane tag="YOLOv11s" variant="yolo" data={yolo} fileName={file.name} />}
          {frcnn && <ModelPane tag="Faster R-CNN" variant="frcnn" data={frcnn} fileName={file.name} />}
        </div>
      </section>
    )
  }

  const image = result?.image || preview
  return (
    <section>
      <div className="card-head">
        <h2>{file.name}</h2>
        {typeof result?.latency_ms === 'number' && (
          <span className="latency-pill">{result.latency_ms.toFixed(1)} ms</span>
        )}
        <button type="button" className="remove" onClick={onRemove}
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
}
