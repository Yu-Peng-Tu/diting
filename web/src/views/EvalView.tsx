import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { PrCurve, Summary, ThresholdCurve, ThresholdPoint } from '../types'
import { ScopePanel } from '../components/ScopePanel'

interface Props {
  summary: Summary
  pr: PrCurve
  thr: ThresholdCurve
}

/** 阈值滑块的实时重算：只用后端下发的 tp/fp/tn/fn，不取缓存指标、不写死数字。 */
function recompute(p: ThresholdPoint) {
  const recall = p.tp + p.fn > 0 ? p.tp / (p.tp + p.fn) : 0
  const fpr = p.fp + p.tn > 0 ? p.fp / (p.fp + p.tn) : 0
  const precision = p.tp + p.fp > 0 ? p.tp / (p.tp + p.fp) : 0
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
  return { recall, fpr, precision, f1 }
}

export function EvalView({ summary, pr, thr }: Props) {
  const binary = summary.same_distribution.binary_at_calibrated_threshold
  const multi = summary.same_distribution.multiclass
  const selectedIdx = useMemo(() => {
    const i = thr.curve.findIndex((p) => p.threshold === thr.selected_threshold)
    return i >= 0 ? i : 0
  }, [thr])
  const [idx, setIdx] = useState(selectedIdx)

  const point = thr.curve[idx]
  const rec = recompute(point)

  // 6917 点全量由后端下发；前端等步长抽样只影响绘制密度，不改变数值
  const prShown = useMemo(() => {
    const step = Math.max(1, Math.ceil(pr.points.length / 600))
    return pr.points.filter((_, i) => i % step === 0 || i === pr.points.length - 1)
  }, [pr.points])

  const cm = multi.confusion_matrix
  const cmMax = Math.max(...cm.matrix.flat(), 1)
  const thrChart = thr.curve.map((p) => ({
    threshold: p.threshold,
    recall: p.recall,
    fpr: p.fpr,
    youden_j: p.youden_j,
  }))

  return (
    <div>
      <div className="card">
        <h2>同分布评估 · HCRL split-v3 val</h2>
        <p className="desc">
          全部数字来自后端 <code>/api/metrics/summary</code> 读取的既有 artifacts JSON。
          窗口级、真实数据子集口径，类别不平衡，不以 accuracy 为主指标。
        </p>
        <div className="grid c4">
          <div className="metric accent">
            <div className="label">F1 @冻结阈值 {summary.frozen_threshold}</div>
            <div className="value">{binary.f1.toFixed(6)}</div>
            <div className="hint">真实数据子集 / split-v3 val / 窗口级</div>
          </div>
          <div className="metric">
            <div className="label">Recall</div>
            <div className="value">{binary.recall.toFixed(6)}</div>
            <div className="hint">TP={binary.tp} · FN={binary.fn}</div>
          </div>
          <div className="metric">
            <div className="label">FPR</div>
            <div className="value">{binary.fpr.toFixed(6)}</div>
            <div className="hint">FP={binary.fp} · TN={binary.tn}，非零误报</div>
          </div>
          <div className="metric">
            <div className="label">Average Precision</div>
            <div className="value">{binary.average_precision.toFixed(6)}</div>
            <div className="hint">PR 曲线面积（阈值无关）</div>
          </div>
          <div className="metric">
            <div className="label">Precision</div>
            <div className="value">{binary.precision.toFixed(6)}</div>
            <div className="hint">每万正常窗口误报 {binary.false_alarms_per_10k_normal}</div>
          </div>
          <div className="metric">
            <div className="label">val 窗口数</div>
            <div className="value">{binary.n_samples.toLocaleString()}</div>
            <div className="hint">正类 {binary.n_positive.toLocaleString()} / 负类 {binary.n_negative.toLocaleString()}</div>
          </div>
          <div className="metric">
            <div className="label">五分类 macro-F1</div>
            <div className="value">{multi.macro_f1.toFixed(6)}</div>
            <div className="hint">仅同分布 val，跨域不可定义</div>
          </div>
          <div className="metric">
            <div className="label">L0 规则基线 F1（同口径重算）</div>
            <div className="value">
              {summary.same_distribution.vs_l0_rows[0]?.f1?.toFixed(6) ?? '—'}
            </div>
            <div className="hint">对照主行用 split-v3 train 重新 fit 阈值</div>
          </div>
        </div>
      </div>

      <div className="grid c2">
        <div className="card">
          <h2>五分类混淆矩阵</h2>
          <p className="desc">行=真实类别，列=预测类别；颜色深浅按单元格数值归一化。</p>
          <table className="data heatmap">
            <thead>
              <tr>
                <th>真实 \ 预测</th>
                {cm.labels.map((l) => (
                  <th key={l}>{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.matrix.map((row, i) => (
                <tr key={cm.labels[i]}>
                  <th>{cm.labels[i]}</th>
                  {row.map((v, j) => {
                    const alpha = v / cmMax
                    const bg = `rgba(37, 99, 235, ${0.06 + alpha * 0.72})`
                    const color = alpha > 0.55 ? '#fff' : 'var(--text)'
                    return (
                      <td key={j} style={{ background: bg, color }}>
                        {v.toLocaleString()}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="grid c2" style={{ marginTop: 14 }}>
            {Object.entries(multi.per_class).map(([cls, m]) => (
              <div className="metric" key={cls}>
                <div className="label">{cls} · support {m.support.toLocaleString()}</div>
                <div className="value" style={{ fontSize: 17 }}>F1 {m.f1.toFixed(6)}</div>
                <div className="hint">P {m.precision.toFixed(6)} · R {m.recall.toFixed(6)}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h2>PR 曲线</h2>
          <p className="desc">
            后端下发 {pr.n_points.toLocaleString()} 个逐点数据，前端等步长绘制 {prShown.length} 点
            （仅降采样显示密度，数值未改写）。AP={pr.average_precision.toFixed(7)}。
          </p>
          <div className="chart-box tall">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prShown} margin={{ left: 4, right: 18, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="recall"
                  type="number"
                  domain={[0, 1]}
                  fontSize={11}
                  label={{ value: 'Recall', position: 'insideBottom', offset: -2, fontSize: 11 }}
                />
                <YAxis domain={[0, 1.02]} fontSize={11} />
                <Tooltip
                  formatter={(v: number, n: string) => [Number(v).toFixed(6), n]}
                  labelFormatter={(l) => `Recall ${Number(l).toFixed(6)}`}
                />
                <Legend />
                <Line isAnimationActive={false} type="monotone" dataKey="precision" name="Precision" stroke="#0284c7" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="desc" style={{ marginTop: 8 }}>{pr.threshold_note}</div>
        </div>
      </div>

      <div className="card">
        <h2>阈值滑块 · 实时重算</h2>
        <p className="desc">
          拖动滑块选取阈值曲线上的网格点，前端用该点的 tp/fp/tn/fn <strong>实时重算</strong>{' '}
          Recall / FPR / Precision / F1（不读取缓存指标）。{thr.recompute_note}
        </p>
        <div className="slider-row">
          <span className="badge gray">网格 {idx + 1} / {thr.n_grid_points}</span>
          <input
            type="range"
            min={0}
            max={thr.n_grid_points - 1}
            step={1}
            value={idx}
            onChange={(e) => setIdx(Number(e.target.value))}
          />
          <span className="thr-val">{point.threshold.toFixed(2)}</span>
          <button className="ghost" onClick={() => setIdx(selectedIdx)}>
            回到冻结阈值 {thr.selected_threshold}
          </button>
        </div>

        <div className="grid c4" style={{ marginTop: 8 }}>
          <div className="metric accent">
            <div className="label">Recall（实时重算）</div>
            <div className="value">{rec.recall.toFixed(6)}</div>
            <div className="hint">tp={point.tp} fn={point.fn}</div>
          </div>
          <div className="metric">
            <div className="label">FPR（实时重算）</div>
            <div className="value">{rec.fpr.toFixed(6)}</div>
            <div className="hint">fp={point.fp} tn={point.tn}</div>
          </div>
          <div className="metric">
            <div className="label">Precision（实时重算）</div>
            <div className="value">{rec.precision.toFixed(6)}</div>
            <div className="hint">由 tp / (tp+fp) 计算</div>
          </div>
          <div className="metric">
            <div className="label">F1（实时重算）</div>
            <div className="value">{rec.f1.toFixed(6)}</div>
            <div className="hint">Youden's J = {point.youden_j.toFixed(6)}</div>
          </div>
        </div>

        <div className="chart-box" style={{ marginTop: 18 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={thrChart} margin={{ left: 4, right: 18, top: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="threshold" fontSize={11} tickFormatter={(v: number) => v.toFixed(2)} />
              <YAxis domain={[0, 1.02]} fontSize={11} />
              <Tooltip formatter={(v: number, n: string) => [Number(v).toFixed(6), n]} />
              <Legend />
              <ReferenceLine
                x={point.threshold}
                stroke="#c92a2a"
                strokeDasharray="4 3"
                label={{ value: '当前点', fontSize: 11, fill: '#c92a2a' }}
              />
              <ReferenceLine
                x={thr.selected_threshold}
                stroke="#08783f"
                strokeDasharray="4 3"
                label={{ value: '冻结阈值', fontSize: 11, fill: '#08783f' }}
              />
              <Line isAnimationActive={false} type="monotone" dataKey="recall" name="Recall" stroke="#0284c7" dot={false} />
              <Line isAnimationActive={false} type="monotone" dataKey="fpr" name="FPR" stroke="#9a5b00" dot={false} />
              <Line isAnimationActive={false} type="monotone" dataKey="youden_j" name="Youden J" stroke="#6d28d9" dot={false} strokeDasharray="5 3" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="notice">
          阈值标定准则：{thr.criterion}（{thr.criterion_note}），在 {thr.calibrated_on} 上取最大；
          选中阈值 {thr.selected_threshold} 对应 J={thr.selected_youden_j.toFixed(6)}。
          此处滑块仅用于演示口径，不改变任何评估结果。
        </div>
        <ScopePanel env={thr} title="模型评估 / 阈值曲线" />
      </div>
    </div>
  )
}
