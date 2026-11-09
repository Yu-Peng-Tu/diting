import type { Envelope } from '../types'

const SCOPE_LABELS: Record<string, string> = {
  evaluated_on: '评估对象',
  data_scope: '数据口径',
  data_scope_short: '口径简写',
  level: '评估粒度',
  split_version: '划分版本',
  dataset: '数据集',
  test_set_status: 'test 折状态',
  source_tag: '来源标记',
  label_policy: '标签口径',
  threshold: '阈值',
}

/** 每个视图底部必挂的口径声明：scope 字段 + 红线提示 + 产物溯源。 */
export function ScopePanel({ env, title }: { env: Envelope; title?: string }) {
  const entries = Object.entries(env.scope || {}).filter(
    ([, v]) => v !== null && v !== undefined && v !== '',
  )
  return (
    <div className="scope">
      <div className="scope-title">口径声明 · {title || '本视图'}</div>
      <dl>
        {entries.map(([k, v]) => (
          <div key={k} style={{ display: 'contents' }}>
            <dt>{SCOPE_LABELS[k] || k}</dt>
            <dd>{String(v)}</dd>
          </div>
        ))}
        <dt>test 载入行数</dt>
        <dd>{env.test_rows_loaded}（封存校验：{env.test_set_status}）</dd>
      </dl>
      <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginTop: 6 }}>数字口径红线</div>
      <ul>
        {(env.caveats || []).map((c, i) => (
          <li key={i}>{c}</li>
        ))}
      </ul>
      <div className="sources">
        数据来源：{Object.values(env.sources || {}).join(' · ') || '见后端白名单产物'}
      </div>
    </div>
  )
}
