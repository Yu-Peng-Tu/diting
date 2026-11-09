interface Props {
  theta: number
  band: number[]
  records: Array<Record<string, any>>
}

/** 边界带标尺：相同位置的错误聚合计数，FP / FN 分行，避免标注重叠。 */
export function BoundaryRuler({ theta, band, records }: Props) {
  const X = (r: number) => 30 + r * 740 // 0..1 → 30..770
  const grouped = new Map<string, { ratio: number; error_kind: string; count: number }>()
  for (const record of records) {
    const ratio = Number(record.attack_ratio)
    const kind = String(record.error_kind)
    const key = `${kind}:${ratio}`
    const current = grouped.get(key)
    grouped.set(key, { ratio, error_kind: kind, count: (current?.count ?? 0) + 1 })
  }
  const markers = [...grouped.values()].sort((a, b) => a.ratio - b.ratio)

  return (
    <svg viewBox="0 0 800 200" className="ruler" role="img" aria-label="attack_ratio 边界带标尺">
      {/* θ 与边界带 */}
      <text x={X(theta)} y="20" textAnchor="middle" className="ruler-theta">θ = {theta}</text>
      <text x={X((band[0] + band[1]) / 2)} y="46" textAnchor="middle" className="ruler-band">
        边界带 [{band[0].toFixed(2)}, {band[1].toFixed(2)}]
      </text>
      <rect x={X(band[0])} y="56" width={X(band[1]) - X(band[0])} height="66"
        fill="var(--amber)" opacity="0.10" />
      <rect x={X(band[0])} y="56" width={X(band[1]) - X(band[0])} height="66"
        fill="none" stroke="var(--amber)" strokeWidth="1" strokeDasharray="3 3" opacity="0.55" />
      <line x1={X(theta)} y1="28" x2={X(theta)} y2="134" stroke="var(--cyan)" strokeWidth="2" opacity="0.8" />

      {/* FP 上行、FN 下行；同一 ratio / 类型只画一个标记并显示数量。 */}
      {markers.map((marker) => {
        const fp = marker.error_kind === 'false_positive'
        const x = X(marker.ratio)
        const label = `${fp ? 'FP' : 'FN'}${marker.count > 1 ? ` ×${marker.count}` : ''}`
        return (
          <g key={`${marker.error_kind}-${marker.ratio}`} className="ruler-marker">
            {fp ? (
              <>
                <text x={x} y="70" textAnchor="middle" className="ruler-fp" fill="var(--red)">{label}</text>
                <path d={`M ${x} 76 l 7 11 h -14 Z`} fill="var(--red)" />
              </>
            ) : (
              <>
                <text x={x} y="99" textAnchor="middle" className="ruler-fp" fill="var(--amber)">{label}</text>
                <rect x={x - 6} y="105" width="12" height="12" fill="var(--amber)"
                  transform={`rotate(45 ${x} 111)`} />
              </>
            )}
          </g>
        )
      })}

      {/* 主尺与说明留出独立行，不与标记共用垂直空间。 */}
      <line x1={X(0)} y1="132" x2={X(1)} y2="132" stroke="var(--hairline)" strokeWidth="2" />
      {Array.from({ length: 21 }, (_, i) => {
        const t = i / 20
        const major = i % 5 === 0
        return (
          <g key={t}>
            <line x1={X(t)} y1={major ? 121 : 126} x2={X(t)} y2="132"
              stroke="var(--text-dim)" strokeWidth={major ? 1.4 : 0.7} />
            {major && <text x={X(t)} y="153" textAnchor="middle" className="ruler-tick">{t.toFixed(2)}</text>}
          </g>
        )
      })}
      <text x={X(theta) + 10} y="174" className="ruler-note">≥ θ 标为攻击窗口</text>
      <text x={X(0)} y="192" className="ruler-axis">attack_ratio（窗口内注入帧占比）</text>
      <text x={X(1)} y="192" textAnchor="end" className="ruler-axis">▲ FP 误报 · ◆ FN 漏报</text>
    </svg>
  )
}
