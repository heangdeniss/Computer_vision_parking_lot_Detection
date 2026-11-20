import { LABELS } from '../lib/constants'

export default function DetectionTable({ detections }) {
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
