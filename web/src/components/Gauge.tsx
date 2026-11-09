interface Props {
  value: number // 0..1 攻击概率
  threshold: number
  decision: 'attack' | 'normal'
}

/** 完整 360° 概率圆环：外圈刻度、中心读数、独立阈值行，无开口圆弧与内部数字。 */
export function Gauge({ value, threshold, decision }: Props) {
  const cx = 130
  const cy = 115
  const radius = 80
  const circumference = 2 * Math.PI * radius
  const v = Math.max(0, Math.min(1, value))
  const color = decision === 'attack' ? 'var(--red)' : 'var(--mint)'
  const point = (t: number, r: number) => {
    const angle = t * 2 * Math.PI - Math.PI / 2
    return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)]
  }
  const [valueX, valueY] = point(v, radius)
  const [thresholdX0, thresholdY0] = point(threshold, radius - 7)
  const [thresholdX1, thresholdY1] = point(threshold, radius + 7)

  return (
    <svg
      viewBox="0 0 260 226"
      className="gauge"
      role="img"
      aria-label={`攻击概率 ${value.toFixed(4)}，判定阈值 ${threshold}`}
    >
      {/* 完整轨道与进度圆环 */}
      <circle cx={cx} cy={cy} r={radius} fill="none" stroke="var(--hairline)" strokeWidth="10" />
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={`${Math.max(v, 0.001) * circumference} ${circumference}`}
        transform={`rotate(-90 ${cx} ${cy})`}
        className="gauge-fill"
      />

      {/* 仅保留四个方向数字，圆环内部不放任何刻度线。 */}
      <text x={cx} y="14" textAnchor="middle" className="gauge-tick">0 / 1</text>
      <text x="239" y="119" textAnchor="middle" className="gauge-tick">0.25</text>
      <text x={cx} y="218" textAnchor="middle" className="gauge-tick">0.50</text>
      <text x="21" y="119" textAnchor="middle" className="gauge-tick">0.75</text>

      {/* 阈值刻线与指针 */}
      <line
        x1={thresholdX0}
        y1={thresholdY0}
        x2={thresholdX1}
        y2={thresholdY1}
        stroke="var(--cyan)"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <circle cx={valueX} cy={valueY} r="5" fill="var(--panel)" stroke={color} strokeWidth="2.5" />

      {/* 中心信息分行，互不占用刻度空间。 */}
      <text x={cx} y="137" textAnchor="middle" className="gauge-value" fill={color}>
        {value.toFixed(4)}
      </text>
      <text x={cx} y="158" textAnchor="middle" className="gauge-caption">P(ATTACK)</text>
      <text x={cx} y="179" textAnchor="middle" className="gauge-thr">阈值 θ = {threshold}</text>
    </svg>
  )
}
