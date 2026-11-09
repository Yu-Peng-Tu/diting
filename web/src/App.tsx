import { useEffect, useState } from 'react'
import { loadDataset } from './api'
import type { Dataset } from './types'
import { DetectView } from './views/DetectView'
import { EvalView } from './views/EvalView'
import { CrossDomainView } from './views/CrossDomainView'
import { GovernanceView } from './views/GovernanceView'

type Tab = 'detect' | 'eval' | 'xdomain' | 'governance'

const TABS: Array<{ key: Tab; label: string; en: string; icon: JSX.Element }> = [
  {
    key: 'detect', label: '总线示波', en: 'BUS SCOPE',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M2 12h4l3-8 4 16 3-8h6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    key: 'eval', label: '阈值调谐', en: 'TUNING',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 6h16M4 12h16M4 18h16" />
        <circle cx="9" cy="6" r="2.4" fill="var(--panel)" />
        <circle cx="15" cy="12" r="2.4" fill="var(--panel)" />
        <circle cx="7" cy="18" r="2.4" fill="var(--panel)" />
      </svg>
    ),
  },
  {
    key: 'xdomain', label: '跨域对照', en: 'CROSS DOMAIN',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="8" cy="12" r="5.5" />
        <circle cx="16" cy="12" r="5.5" />
      </svg>
    ),
  },
  {
    key: 'governance', label: '事件取证', en: 'FORENSICS',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="M15.5 15.5 21 21" strokeLinecap="round" />
        <path d="M8 10.5h5M10.5 8v5" />
      </svg>
    ),
  },
]

export default function App() {
  const hashTab = window.location.hash.replace('#', '') as Tab
  const [tab, setTab] = useState<Tab>(TABS.some((t) => t.key === hashTab) ? hashTab : 'detect')
  const [data, setData] = useState<Dataset | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    loadDataset()
      .then((d) => setData(d as unknown as Dataset))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  return (
    <div className="console">
      <header className="statusbar">
        <div className="brand">
          <div className="radar"><i /></div>
          <div className="brand-text">
            <div className="brand-name">谛听<span className="brand-en">DITING</span></div>
            <div className="brand-sub">IN-VEHICLE CAN INTRUSION DETECTION CONSOLE</div>
          </div>
        </div>
        <div className="leds">
          <span className="led"><i className="dot cyan" />MODEL {data?.health.model_version ?? '······'}</span>
          <span className="led"><i className="dot cyan" />THR {data?.health.frozen_threshold ?? '—'}</span>
          <span className="led"><i className="dot cyan" />SPLIT v3</span>
          <span className="led"><i className="dot green" />TEST SEALED · {data?.health.test_rows_loaded ?? '—'} ROWS</span>
          <span className="led"><i className="dot amber" />子集 / VAL / 零样本 / 适配</span>
        </div>
      </header>

      <div className="deck">
        <nav className="rail">
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`rail-btn ${tab === t.key ? 'active' : ''}`}
              onClick={() => setTab(t.key)}
            >
              <span className="rail-icon">{t.icon}</span>
              <span className="rail-label">{t.label}</span>
              <span className="rail-en">{t.en}</span>
            </button>
          ))}
          <div className="rail-foot">L1-v3<br />GBDT</div>
        </nav>

        <main className="workspace">
          {error && (
            <div className="error">
              <strong>无法加载后端数据</strong>
              <br />
              {error}
              <br />
              请先启动后端：<code>uvicorn diting.server.app:app --host 127.0.0.1 --port 8765</code>
            </div>
          )}

          {!data && !error && <div className="loading">正在从后端加载评估产物…</div>}

          {data && tab === 'detect' && (
            <DetectView samples={data.samples} frozenThreshold={data.summary.frozen_threshold} />
          )}
          {data && tab === 'eval' && (
            <EvalView summary={data.summary} pr={data.pr} thr={data.thr} />
          )}
          {data && tab === 'xdomain' && (
            <CrossDomainView summary={data.summary} ablation={data.ablation} />
          )}
          {data && tab === 'governance' && (
            <GovernanceView errors={data.errors} summary={data.summary} />
          )}

          <div className="footer">
            所有展示数字来自后端真实产物 JSON，前端不写死指标 · HCRL test 折全程封存 ·
            同分布为真实数据子集 split-v3 val 窗口级结果，跨域为 OTIDS（Kia Soul）零样本 / 阈值适配口径，不可混报
          </div>
        </main>
      </div>
    </div>
  )
}
