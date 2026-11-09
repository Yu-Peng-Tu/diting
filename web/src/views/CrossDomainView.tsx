import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { Ablation, Summary } from '../types'
import { ScopePanel } from '../components/ScopePanel'

interface Props {
  summary: Summary
  ablation: Ablation
}

const COLORS = ['#0284c7', '#c92a2a', '#08783f', '#6d28d9']

export function CrossDomainView({ summary, ablation }: Props) {
  const sd = summary.same_distribution.binary_at_calibrated_threshold
  const zero = summary.cross_domain.zeroshot_at_frozen_0_54
  const zero05 = summary.cross_domain.zeroshot_at_0_5
  const rows = summary.cross_domain.adaptation_rows
  const adapted = rows.find((r) => r.policy === 'otids_adapted')
  const zeroOnEval = rows.find((r) => r.policy === 'otids_zeroshot_on_eval_fold')
  const official = summary.cross_domain.official_label_sensitivity
  const officialRow = (official?.rows || []).find((r: any) => r.tag === '0_54')

  const columns = [
    {
      title: '同分布 val（HCRL）',
      tag: '子集 / split-v3 val / 窗口级',
      metrics: sd,
      color: COLORS[0],
      note: `训练同分布，阈值 ${summary.frozen_threshold} 在 val 上标定`,
    },
    {
      title: '跨车型零样本（OTIDS）',
      tag: `零样本 / 冻结阈值 ${summary.frozen_threshold} / 窗口级`,
      metrics: zero,
      color: COLORS[1],
      note: '模型权重、归一化、ID 频次表、阈值全部冻结不改',
    },
    {
      title: '阈值适配上限',
      tag: '非零样本 / 仅调阈值 / eval 折',
      metrics: adapted?.metrics,
      color: COLORS[2],
      note: adapted
        ? `在 OTIDS adapt 折重标定阈值至 ${adapted.threshold}；不得与零样本混报`
        : '产物缺失',
    },
  ]

  const xTypes = summary.cross_domain.per_attack_type
  const sdTypes = summary.same_distribution.per_attack_type
  const recallRows = ['DoS', 'Fuzzy', 'Impersonation'].map((t) => ({
    type: t,
    otids: Number((xTypes[t]?.recall_at_frozen ?? 0) as number),
    hcrl: t === 'Impersonation' ? null : Number((sdTypes[t]?.recall ?? 0) as number),
    support: xTypes[t]?.support ?? 0,
  }))

  const driftRows = Object.entries(summary.cross_domain.feature_drift_by_group).map(
    ([group, v]) => ({ group, mean: v.mean_abs_delta, max: v.max_abs_delta, n: v.n_features }),
  )

  const mapping = summary.cross_domain.five_class_mapping
  const crossAbl = ablation.cross_domain

  return (
    <div>
      <div className="card">
        <h2>同分布 vs 跨车型零样本 vs 阈值适配上限</h2>
        <p className="desc">
          三列口径互不相同，<strong>严禁混报或取更优者</strong>。数字全部来自后端读取的
          <code>xdomain_*</code> 与 <code>l1_v3_*</code> 产物。
        </p>
        <div className="grid c3">
          {columns.map((c) => (
            <div className="metric" key={c.title} style={{ borderTop: `3px solid ${c.color}` }}>
              <div className="label" style={{ fontWeight: 700, color: c.color, fontSize: 13 }}>
                {c.title}
              </div>
              <div className="value" style={{ fontSize: 26 }}>
                {c.metrics ? c.metrics.f1.toFixed(6) : '—'}
              </div>
              <div className="hint" style={{ marginBottom: 8 }}>F1</div>
              <div className="tag-list"><span className="badge blue">{c.tag}</span></div>
              <dl className="kv" style={{ marginTop: 12 }}>
                <dt>Recall</dt><dd>{c.metrics ? c.metrics.recall.toFixed(6) : '—'}</dd>
                <dt>Precision</dt><dd>{c.metrics ? c.metrics.precision.toFixed(6) : '—'}</dd>
                <dt>FPR</dt><dd>{c.metrics ? c.metrics.fpr.toFixed(6) : '—'}</dd>
                <dt>AP</dt><dd>{c.metrics ? c.metrics.average_precision.toFixed(8) : '—'}</dd>
                <dt>窗口数</dt><dd>{c.metrics ? c.metrics.n_samples.toLocaleString() : '—'}</dd>
              </dl>
              <div className="hint" style={{ marginTop: 10 }}>{c.note}</div>
            </div>
          ))}
        </div>

        <div className="grid c3" style={{ marginTop: 16 }}>
          <div className="metric">
            <div className="label">零样本 @0.5（未标定阈值）</div>
            <div className="value" style={{ fontSize: 20 }}>{zero05.f1.toFixed(6)}</div>
            <div className="hint">Recall {zero05.recall.toFixed(6)} · AP {zero05.average_precision.toFixed(8)}</div>
          </div>
          <div className="metric">
            <div className="label">同一 eval 折 · 冻结阈值（隔离阈值贡献）</div>
            <div className="value" style={{ fontSize: 20 }}>
              {zeroOnEval ? zeroOnEval.metrics.f1.toFixed(6) : '—'}
            </div>
            <div className="hint">
              与适配上限在同一 eval 折，F1 差 {summary.cross_domain.same_eval_fold_delta_vs_frozen.f1_delta?.toFixed(6)}
            </div>
          </div>
          <div className="metric">
            <div className="label">官方描述口径敏感性（@{summary.frozen_threshold}）</div>
            <div className="value" style={{ fontSize: 20 }}>
              {officialRow ? Number(officialRow.fpr).toFixed(6) : '见产物 JSON'}
            </div>
            <div className="hint">
              FPR（标签口径换成官方描述后）；同口径 F1 {officialRow ? Number(officialRow.f1).toFixed(6) : '—'}，
              说明主口径 FPR={zero.fpr.toFixed(6)} 不是零误报
            </div>
          </div>
        </div>

        <div className="notice danger">
          <strong>禁止把 FPR=0 当作「零误报」卖点。</strong>
          主口径是文件级标签，负类窗口全部来自独立 Attack-Free 文件（会话级判定）；
          换成官方描述口径 FPR 立刻不同。同理，适配上限使用了一部分 OTIDS 数据调阈值，
          <strong>不是零样本泛化能力</strong>，不得与零样本数字混报或取更优值对外引用。
        </div>
      </div>

      <div className="grid c2">
        <div className="card">
          <h2>分攻击类型 Recall（@冻结阈值）</h2>
          <p className="desc">
            OTIDS 三类来自跨域产物；HCRL 同分布列仅作参考（Impersonation 在 HCRL 无对应类，留空不补数）。
          </p>
          <div className="chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={recallRows} margin={{ left: 4, right: 16 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" fontSize={11} />
                <YAxis domain={[0, 1.05]} fontSize={11} />
                <Tooltip formatter={(v: number, n: string) => [Number(v).toFixed(6), n]} />
                <Legend />
                <Bar isAnimationActive={false} dataKey="otids" name="OTIDS 零样本 Recall" fill="#c92a2a" radius={[3, 3, 0, 0]} />
                <Bar isAnimationActive={false} dataKey="hcrl" name="HCRL 同分布 Recall" fill="#0284c7" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="data" style={{ marginTop: 12 }}>
            <thead><tr><th>类型</th><th className="num">OTIDS support</th><th className="num">Recall@frozen</th><th className="num">漏检窗口</th></tr></thead>
            <tbody>
              {recallRows.map((r) => (
                <tr key={r.type}>
                  <td>{r.type}</td>
                  <td className="num">{Number(r.support).toLocaleString()}</td>
                  <td className="num">{r.otids.toFixed(6)}</td>
                  <td className="num">{xTypes[r.type]?.missed_at_frozen ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>特征分布漂移（归一化空间 |Δmean|）</h2>
          <p className="desc">
            HCRL val 与 OTIDS 窗口的 33 维特征按组统计；单位≈1 个 HCRL train std。
            最大漂移组：<strong>{summary.cross_domain.dominant_drift_group}</strong>。
            漂移幅度大 ≠ 对 F1 下降的贡献大，本视图不编造贡献占比。
          </p>
          <div className="chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={driftRows} layout="vertical" margin={{ left: 16, right: 26 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" fontSize={11} />
                <YAxis type="category" dataKey="group" width={90} fontSize={11} />
                <Tooltip formatter={(v: number, n: string) => [Number(v).toFixed(6), n]} />
                <Legend />
                <Bar isAnimationActive={false} dataKey="mean" name="组平均 |Δmean|" radius={[0, 3, 3, 0]}>
                  {driftRows.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="data" style={{ marginTop: 12 }}>
            <thead><tr><th>组</th><th className="num">维度数</th><th className="num">平均 |Δmean|</th><th className="num">最大 |Δmean|</th></tr></thead>
            <tbody>
              {driftRows.map((d) => (
                <tr key={d.group}>
                  <td>{d.group}</td><td className="num">{d.n}</td>
                  <td className="num">{d.mean.toFixed(6)}</td><td className="num">{d.max.toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid c2">
        <div className="card">
          <h2>五分类映射（次级口径，非准确率）</h2>
          <div className="notice">
            <strong>跨域 macro-F1 不可定义，本工作台不报出。</strong>
            macro_f1_reported={String(mapping.macro_f1_reported)}；原因：{mapping.macro_f1_reason}
          </div>
          <table className="data">
            <thead><tr><th>OTIDS 真值类型</th><th className="num">窗口数</th><th>模型预测分布（%）</th></tr></thead>
            <tbody>
              {Object.entries(mapping.per_true_type).map(([t, v]: [string, any]) => (
                <tr key={t}>
                  <td>{t}</td>
                  <td className="num">{Number(v.support).toLocaleString()}</td>
                  <td>
                    {Object.entries(v.predicted_distribution_pct || {})
                      .filter(([, pct]) => Number(pct) > 0)
                      .map(([cls, pct]) => `${cls} ${Number(pct).toFixed(2)}%`)
                      .join(' · ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="desc" style={{ marginTop: 10 }}>
            映射分布仅为定性参考：OTIDS 无 Gear/RPM 真值、Impersonation 无对应 HCRL 类别，
            强行计算五分类准确率会造成口径误导。
          </p>
        </div>

        <div className="card">
          <h2>跨域「去 ID 频率组」消融</h2>
          <p className="desc">{crossAbl.collapse_rule} · 训练来源：{crossAbl.trained_on}</p>
          <div className="grid c2">
            <div className="metric">
              <div className="label">全特征 33 维 AP</div>
              <div className="value" style={{ fontSize: 20 }}>{Number(crossAbl.ap_full_33).toFixed(8)}</div>
            </div>
            <div className="metric">
              <div className="label">去 ID 后 AP</div>
              <div className="value" style={{ fontSize: 20 }}>{Number(crossAbl.ap_drop_id).toFixed(8)}</div>
            </div>
            <div className="metric">
              <div className="label">AP 相对变化</div>
              <div className="value" style={{ fontSize: 20 }}>{Number(crossAbl.ap_relative_drop_pct).toFixed(4)}%</div>
            </div>
            <div className="metric">
              <div className="label">是否崩塌</div>
              <div className="value" style={{ fontSize: 20 }}>{crossAbl.collapsed ? '是' : '否'}</div>
            </div>
          </div>
          <div className="notice">
            {crossAbl.conclusion}
            <br />
            <strong>{ablation.direction_note}</strong>
          </div>
          <ScopePanel env={ablation} title="跨域消融" />
        </div>
      </div>

      <div className="card">
        <ScopePanel env={summary} title="跨域验证" />
      </div>
    </div>
  )
}
