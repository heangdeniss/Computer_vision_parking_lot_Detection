import { STRIDES } from '../lib/constants'

export default function Controls({
  tab, conf, stride, realtime, live, loading, canDetect,
  onConf, onStride, onRealtime, onDetect, onClear,
}) {
  return (
    <div className="controls">
      <label className="slider">
        Confidence <strong>{conf.toFixed(2)}</strong>
        <input type="range" min="0.05" max="0.9" step="0.05" value={conf}
          onChange={(e) => onConf(Number(e.target.value))} />
      </label>
      {tab === 'video' && (
        <label className="check">
          <input type="checkbox" checked={realtime} onChange={(e) => onRealtime(e.target.checked)} />
          Realtime (live boxes)
        </label>
      )}
      {tab === 'video' && !realtime && (
        <label className="field">
          Sample every
          <select value={stride} onChange={(e) => onStride(Number(e.target.value))}>
            {STRIDES.map((n) => <option key={n} value={n}>{n} frames</option>)}
          </select>
        </label>
      )}
      <button type="submit" disabled={loading || !canDetect}>
        {loading ? 'Detecting…' : tab === 'video' && realtime ? (live ? 'Stop live' : 'Start live') : 'Detect'}
      </button>
      <button type="button" className="secondary" onClick={onClear}>Clear</button>
    </div>
  )
}
