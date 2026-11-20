import { LABELS } from '../lib/constants'

export default function Badges({ counts }) {
  return (
    <div className="counts">
      {Object.entries(counts).map(([name, count]) => (
        <span key={name} className={`badge ${name}`}>{LABELS[name] || name}: {count}</span>
      ))}
    </div>
  )
}
