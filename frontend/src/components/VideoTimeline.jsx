import Badges from './Badges'
import DetectionTable from './DetectionTable'

export default function VideoTimeline({ result, frame, onSelect }) {
  const selected = result.results[frame]
  return (
    <section className="video-result">
      <h2>Frame timeline — {result.name}</h2>
      <p className="meta">
        {result.sampled} sampled of {result.total_frames} frames · {result.fps} fps · every {result.stride} frames
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
