import Badges from './Badges'

export default function LiveView({ name, canvasRef, stats }) {
  return (
    <section className="video-result">
      <h2>Live detection — {name}</h2>
      <div className="live-stage"><canvas ref={canvasRef} /></div>
      <p className="meta">{stats ? `${stats.fps} fps · live inference` : 'Starting live stream…'}</p>
      {stats && <Badges counts={stats.counts} />}
    </section>
  )
}
