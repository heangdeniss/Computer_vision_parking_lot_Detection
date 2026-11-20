import Badges from './Badges'
import DetectionTable from './DetectionTable'

export default function ImageCard({ file, preview, result, onRemove }) {
  const image = result?.image || preview
  return (
    <section>
      <div className="card-head">
        <h2>{file.name}</h2>
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
