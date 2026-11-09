import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { postDetect } from '../api'
import type { DetectResult, Samples } from '../types'
import { ScopePanel } from '../components/ScopePanel'
import { Gauge } from '../components/Gauge'

const EXAMPLE_HINT =
  '粘贴 candump 文本，每行一帧，例如：\n(1479121434.850202) can0 316#052100090021006F\n也支持 CSV：timestamp,can_id,dlc,8字节数据。至少 8 帧，超过 100 帧取最后 100 帧。'

interface Props {
  samples: Samples
  frozenThreshold: number
}

export function DetectView({ samples, frozenThreshold }: Props) {
  const [mode, setMode] = useState<'sample' | 'candump'>('sample')
  const [sampleId, setSampleId] = useState((samples.samples.find((s) => s.label === 1) ?? samples.samples[0])?.sample_id || '')
  const [candump, setCandump] = useState('')
  const [thresholdText, setThresholdText] = useState('')
  const [result, setResult] = useState<DetectResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const first = useRef(true)

  const grouped = useMemo(() => {
    const map: Record<string, typeof samples.samples> = {}
    for (const s of samples.samples) {
      const key = s.source === 'hcrl_val' ? 'HCRL VAL 通道' : 'OTIDS 跨域通道'
      ;(map[key] ||= []).push(s)
    }
    return map
  }, [samples.samples])

  const run = async (overrideId?: string) => {
    setBusy(true)
    setError('')
    try {
      const body: { sample_id?: string; candump?: string; threshold?: number } = {}
      if (mode === 'sample' || overrideId) body.sample_id = overrideId ?? sampleId
      else body.candump = candump
      if (thresholdText.trim() !== '') {
        const t = Number(thresholdText)
        if (!Number.isFinite(t) || t < 0 || t > 1) throw new Error('阈值需在 [0, 1] 区间内')
        body.threshold = t
      }
      setResult(await postDetect(body))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  // 选中通道即推理；挂载后自动跑第一个样本，让仪器上电即有读数
  useEffect(() => {
    if (first.current && sampleId) {
      first.current = false
      void run(sampleId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const pick = (id: string) => {
    setSampleId(id)
    void run(id)
  }

  const contributionData = (result?.top_features || []).map((f) => ({
    name: f.feature,
    group: f.group,
    value: Number(f.contribution_to_attack_proba.toFixed(6)),
  }))
  const multiData = Object.entries(result?.multiclass.probabilities || {}).map(([name, value]) => ({
    name,
    value: Number(value.toFixed(6)),
  }))

  return (
    <div className="scope-grid">
      {/* ── 左：信号通道 ─────────────────────────── */}
      <section className="panel channels">
        <div className="panel-title">信号接入<span className="panel-en">INPUT</span></div>
        <div className="mode-switch">
          <button className={mode === 'sample' ? 'active' : ''} onClick={() => setMode('sample')}>样本通道</button>
          <button className={mode === 'candump' ? 'active' : ''} onClick={() => setMode('candump')}>外部报文</button>
        </div>

        {mode === 'sample' ? (
          <div className="channel-list">
            {Object.entries(grouped).map(([group, items]) => (
              <div key={group}>
                <div className="channel-group">{group}</div>
                {items.map((s) => (
                  <button
                    key={s.sample_id}
                    className={`channel ${sampleId === s.sample_id ? 'active' : ''}`}
                    onClick={() => pick(s.sample_id)}
                  >
                    <span className={`channel-led ${s.label === 1 ? 'red' : 'green'}`} />
                    <span className="channel-name">{s.dominant_class}</span>
                    <span className="channel-meta">r={s.attack_ratio.toFixed(2)}</span>
                    <span className="channel-id">{s.window_id}</span>
                  </button>
                ))}
              </div>
            ))}
          </div>
        ) : (
          <div className="candump-box">
            <textarea
              value={candump}
              placeholder={EXAMPLE_HINT}
              onChange={(e) => setCandump(e.target.value)}
            />
            <button className="primary" disabled={busy} onClick={() => run()}>
              {busy ? '推理中…' : '接入并检测'}
            </button>
          </div>
        )}

        <div className="thr-input">
          <label>判定阈值（留空 = 冻结 {frozenThreshold}）</label>
          <input
            type="number" step="0.01" min="0" max="1"
            value={thresholdText}
            placeholder={`${frozenThreshold}`}
            onChange={(e) => setThresholdText(e.target.value)}
          />
        </div>
        <div className="channel-note">
          样本只来自 split-v3 val 与 OTIDS；test 折全程封存。点选通道即做真实推理（非 mock）。
        </div>
      </section>

      {/* ── 中：示波表头 ─────────────────────────── */}
      <section className="panel scope-main">
        <div className="panel-title">攻击概率示波<span className="panel-en">P(ATTACK)</span></div>
        {error && <div className="notice danger">请求失败：{error}</div>}
        {busy && <div className="scope-busy">采样推理中…</div>}
        {result ? (
          <>
            <div className={`verdict ${result.decision}`}>
              <span className="verdict-led" />
              {result.decision === 'attack' ? '攻击窗口 · ATTACK' : '正常窗口 · NORMAL'}
              <span className="verdict-thr">
                {result.threshold_is_frozen ? `冻结阈值 ${result.threshold}` : `演示阈值 ${result.threshold}`}
              </span>
            </div>
            <Gauge value={result.attack_probability} threshold={result.threshold} decision={result.decision} />
            <div className="scope-facts">
              <div><span>五分类预测</span><strong>{result.multiclass.predicted_class}</strong></div>
              <div><span>P(normal)</span><strong>{result.normal_probability.toFixed(4)}</strong></div>
              <div><span>输入</span><strong>{result.input.n_frames ? `${result.input.n_frames} 帧` : '单窗口'}</strong></div>
            </div>
            <div className="scope-id">{result.input.window_id || 'user:candump'} · {result.model_version} · {result.n_features} 维 G1–G6</div>
            <div className="notice info" style={{ marginTop: 14 }}>
              <strong>特征贡献方法</strong>：{result.contribution_method}
            </div>
          </>
        ) : (
          !busy && <div className="scope-empty">选择左侧通道，或接入外部 candump 报文</div>
        )}
      </section>

      {/* ── 右：信号分析 ─────────────────────────── */}
      <section className="panel analysis">
        <div className="panel-title">信号分析<span className="panel-en">ANALYSIS</span></div>
        {result && (
          <>
            <div className="analysis-sub">五分类概率分布（HCRL 类目，非真值）</div>
            <div className="chart-box" style={{ height: 190 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={multiData} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 1]} fontSize={11} />
                  <YAxis type="category" dataKey="name" width={66} fontSize={11} />
                  <Tooltip formatter={(v: number) => v.toFixed(6)} />
                  <Bar dataKey="value" fill="#087887" radius={[0, 3, 3, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="analysis-sub">Top 8 局部特征贡献 ΔP(attack)</div>
            <div className="chart-box" style={{ height: 230 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={contributionData} layout="vertical" margin={{ left: 8, right: 32 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" fontSize={11} tickFormatter={(v: number) => v.toFixed(3)} />
                  <YAxis type="category" dataKey="name" width={132} fontSize={10} />
                  <Tooltip formatter={(v: number) => v.toFixed(6)} />
                  <Bar dataKey="value" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                    {contributionData.map((d, i) => (
                      <Cell key={i} fill={d.value >= 0 ? '#c92a2a' : '#08783f'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </section>

      <div className="scope-bottom card">
        <ScopePanel env={result ?? samples} title={result ? '实时检测演示' : '演示样本库'} />
      </div>
    </div>
  )
}
