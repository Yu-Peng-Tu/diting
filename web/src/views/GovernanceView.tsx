import type { ErrorsData, Summary } from '../types'
import { ScopePanel } from '../components/ScopePanel'
import { BoundaryRuler } from '../components/BoundaryRuler'

interface Props {
  errors: ErrorsData
  summary: Summary
}

export function GovernanceView({ errors, summary }: Props) {
  const band = errors.label_boundary_analysis.near_boundary_band
  const records = errors.records
  const rows = summary.same_distribution.vs_l0_rows
  const deltas = summary.same_distribution.vs_l0_deltas

  return (
    <div>
      <div className="card">
        <h2>误报治理 · 逐窗归因</h2>
        <p className="desc">
          阈值 {errors.threshold_used}，在 {errors.n_val_windows.toLocaleString()} 个 split-v3 val 窗口上：
          误报 {errors.n_false_positives} 个、漏报 {errors.n_false_negatives} 个。
          下面是 <code>l1_v3_error_attribution.json</code> 的全量逐窗记录，非抽样。
        </p>
        <div className="grid c4">
          <div className="metric danger">
            <div className="label">误报 FP</div>
            <div className="value">{errors.n_false_positives}</div>
            <div className="hint">Normal 窗口被判为攻击</div>
          </div>
          <div className="metric">
            <div className="label">漏报 FN</div>
            <div className="value">{errors.n_false_negatives}</div>
            <div className="hint">攻击窗口未被检出</div>
          </div>
          <div className="metric">
            <div className="label">窗口标签硬阈值 θ</div>
            <div className="value">{errors.label_boundary_analysis.window_label_theta}</div>
            <div className="hint">attack_ratio ≥ θ 标为攻击窗口</div>
          </div>
          <div className="metric accent">
            <div className="label">落在边界带的错误</div>
            <div className="value">
              {errors.label_boundary_analysis.n_errors_near_boundary} / {errors.label_boundary_analysis.n_errors}
            </div>
            <div className="hint">边界带 [{band[0]?.toFixed(2)}, {band[1]?.toFixed(2)}]</div>
          </div>
        </div>

        <div className="notice info" style={{ marginTop: 16 }}>
          <strong>核心结论</strong>：{errors.label_boundary_analysis.finding}
        </div>
        <BoundaryRuler
          theta={errors.label_boundary_analysis.window_label_theta}
          band={band}
          records={records}
        />
        <p className="desc">{errors.note}</p>
      </div>

      <div className="card">
        <h2>错误窗口明细（含 attack_ratio 边界带标注）</h2>
        <p className="desc">
          浅橙底 = attack_ratio 落在 [{band[0]?.toFixed(2)}, {band[1]?.toFixed(2)}] 边界带内。
          score 为冻结模型的攻击概率，非人工填写。
        </p>
        <table className="data">
          <thead>
            <tr>
              <th>window_id</th>
              <th>错误类型</th>
              <th>真值类</th>
              <th className="num">label</th>
              <th className="num">binary_pred</th>
              <th className="num">attack_ratio</th>
              <th className="num">score</th>
              <th>五分类预测</th>
              <th>边界带</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r: any) => {
              const ratio = Number(r.attack_ratio)
              const inBand = ratio >= band[0] && ratio <= band[1]
              return (
                <tr key={String(r.window_id)} className={inBand ? 'boundary' : ''}>
                  <td style={{ fontFamily: 'monospace', fontSize: 11 }}>{String(r.window_id)}</td>
                  <td>
                    <span className={`badge ${r.error_kind === 'false_positive' ? 'red' : 'amber'}`}>
                      {r.error_kind === 'false_positive' ? '误报 FP' : '漏报 FN'}
                    </span>
                  </td>
                  <td>{String(r.attack_type)}</td>
                  <td className="num">{r.label}</td>
                  <td className="num">{r.binary_pred}</td>
                  <td className="num">{ratio.toFixed(2)}</td>
                  <td className="num">{Number(r.score).toFixed(8)}</td>
                  <td>{String(r.multiclass_pred)}</td>
                  <td>{inBand ? '✓ 边界带内' : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>

        <h3>偏差最大的特征（相对同类 val 中位数）</h3>
        <div className="grid c2">
          <div className="metric">
            <div className="label">误报窗口</div>
            <div className="value" style={{ fontSize: 14 }}>
              {(errors.top_deviated_features_in_false_positives || [])
                .slice(0, 6)
                .map((f: any) => f.feature)
                .join(' · ')}
            </div>
          </div>
          <div className="metric">
            <div className="label">漏报窗口</div>
            <div className="value" style={{ fontSize: 14 }}>
              {(errors.top_deviated_features_in_false_negatives || [])
                .slice(0, 6)
                .map((f: any) => f.feature)
                .join(' · ')}
            </div>
          </div>
        </div>
        <ScopePanel env={errors} title="误报治理" />
      </div>

      <div className="grid c2">
        <div className="card">
          <h2>治理建议（产物原文）</h2>
          <p className="desc">按 <code>l1_v3_error_attribution.json</code> 的原始顺序展示，未改写。</p>
          <ol style={{ fontSize: 12.5, lineHeight: 1.85, paddingLeft: 20, margin: 0 }}>
            {errors.governance_recommendations.map((g, i) => (
              <li key={i} style={{ marginBottom: 9 }}>{g}</li>
            ))}
          </ol>
        </div>

        <div className="card">
          <h2>L0 规则基线 vs L1（同一 split-v3 val）</h2>
          <p className="desc">
            主行口径：L0 用 split-v3 train 正常窗口重新 fit 规则阈值，与 L1 训练/评估划分完全一致。
            数字来自 <code>l1_v3_vs_l0.json</code>。
          </p>
          <table className="data">
            <thead><tr><th>模型</th><th className="num">Recall</th><th className="num">FPR</th><th className="num">F1</th><th className="num">阈值</th></tr></thead>
            <tbody>
              {rows.map((r: any, i: number) => (
                <tr key={i}>
                  <td>{String(r.model)}</td>
                  <td className="num">{Number(r.recall).toFixed(6)}</td>
                  <td className="num">{Number(r.fpr).toFixed(6)}</td>
                  <td className="num">{Number(r.f1).toFixed(6)}</td>
                  <td className="num">{String(r.threshold)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>相对 L0 的变化</h3>
          <table className="data">
            <thead><tr><th>对照</th><th className="num">FPR 绝对变化</th><th className="num">FPR 相对变化</th><th className="num">F1 变化</th></tr></thead>
            <tbody>
              {deltas.map((d: any, i: number) => (
                <tr key={i}>
                  <td>{String(d.model)}</td>
                  <td className="num">{Number(d.fpr_absolute_change).toFixed(6)}</td>
                  <td className="num">{Number(d.fpr_relative_change_pct).toFixed(4)}%</td>
                  <td className="num">{Number(d.f1_absolute_change).toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="notice">
            近满分指标受子集口径、窗口级标签与类别不平衡影响，不可外推为逐帧检出率或任意车型表现；
            治理优先级仍以「标签口径治理」为首要项。
          </div>
          <ScopePanel env={summary} title="L0/L1 对照" />
        </div>
      </div>
    </div>
  )
}
