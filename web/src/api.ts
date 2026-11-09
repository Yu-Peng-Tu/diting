import type { DetectResult } from './types'

async function parseResponse<T>(res: Response, path: string): Promise<T> {
  if (!res.ok) {
    let detail = ''
    try {
      detail = JSON.stringify(await res.json())
    } catch {
      detail = await res.text()
    }
    throw new Error(`${path} → HTTP ${res.status} ${detail}`)
  }
  return (await res.json()) as T
}

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: 'application/json' } })
  return parseResponse<T>(res, path)
}

export async function postDetect(body: {
  sample_id?: string
  candump?: string
  threshold?: number
}): Promise<DetectResult> {
  const res = await fetch('/api/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  })
  return parseResponse<DetectResult>(res, '/api/detect')
}

export function loadDataset(): Promise<{
  summary: any
  pr: any
  thr: any
  ablation: any
  errors: any
  samples: any
  health: any
}> {
  return Promise.all([
    getJson('/api/metrics/summary'),
    getJson('/api/curves/pr'),
    getJson('/api/curves/threshold'),
    getJson('/api/ablation'),
    getJson('/api/errors'),
    getJson('/api/samples'),
    getJson('/api/health'),
  ]).then(([summary, pr, thr, ablation, errors, samples, health]) => ({
    summary,
    pr,
    thr,
    ablation,
    errors,
    samples,
    health,
  }))
}
