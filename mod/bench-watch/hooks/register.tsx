import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { Activity } from '../types'

const PANE = 'bench-watch'
const activity = atom({ plugin: 'bench-watch', key: 'activity' } as const, [])

type Row = { label: string; loc: number; cld: number }
type Snapshot = { load: string; jobs: string[]; logs: { name: string; line: string }[]; rows: Row[]; tests: number }

const JOB = /suite_\w+\.py.*|ann_scale\.py|build_report\.py.*|llama-server -m \S+/

async function snapshot($: any, home: string): Promise<Snapshot> {
  const base = (await $.env.get('BENCH_DIR')) ?? `${home}/embeddinggemma2-vs-gemini-bench`
  const load = (await $.process.run(['sysctl', '-n', 'vm.loadavg']).catch(() => ({ stdout: '' }))).stdout.replace(/[{}]/g, '').trim().split(/\s+/)[0] ?? '?'
  const ps = (await $.process.run(['ps', '-Ao', 'etime=,command=']).catch(() => ({ stdout: '' }))).stdout
  const jobs = ps.split('\n').map((l: string) => {
    const m = l.match(JOB)
    return m ? `${l.trim().split(/\s+/)[0]}  ${m[0].replace('gguf/', '').slice(0, 90)}` : ''
  }).filter(Boolean)

  const logs: Snapshot['logs'] = []
  const entries = await $.fs.list(`${base}/logs`).catch(() => [])
  for (const f of [...entries].filter((x: any) => x.name.endsWith('.log')).sort((a: any, b: any) => b.mtimeMs - a.mtimeMs).slice(0, 3)) {
    const text: string = await $.fs.read(`${base}/logs/${f.name}`).catch(() => '')
    const last = text.trimEnd().split('\n').filter((l) => l.trim()).pop() ?? ''
    logs.push({ name: f.name.replace('.log', ''), line: last.slice(0, 160) })
  }

  const j = async (n: string) => JSON.parse(await $.fs.read(`${base}/results/${n}.json`).catch(() => '{}'))
  const [T, M, X] = [await j('text'), await j('media'), await j('more')]
  const rows: Row[] = []
  const add = (label: string, loc?: number, cld?: number) => { if (typeof loc === 'number' && typeof cld === 'number') rows.push({ label, loc, cld }) }
  for (const [k, l] of [['codesearch', 'Code search'], ['stackoverflow', 'Dev Q&A'], ['quora', 'Duplicates'], ['fiqa', 'Finance'], ['legal', 'Legal'], ['scifact', 'Science'], ['nfcorpus', 'Medical']] as const)
    add(l, T[k]?.loc_768?.ndcg10, T[k]?.cloud_768?.ndcg10)
  add('Ticket routing', T.banking77?.loc_768?.accuracy, T.banking77?.cloud_768?.accuracy)
  add('Similarity', T.stsb?.loc_768?.spearman, T.stsb?.cloud_768?.spearman)
  add('HR fit (AUC)', T.resume?.loc_768?.auc, T.resume?.cloud_768?.auc)
  if (T.tatoeba?.langs) {
    const v = Object.values(T.tatoeba.langs) as any[]
    add('Languages', v.reduce((s, x) => s + x.loc, 0) / v.length, v.reduce((s, x) => s + x.cloud, 0) / v.length)
  }
  add('Photos R@1', M.photos?.loc_t2i_768?.['R@1'], M.photos?.cloud_t2i_768?.['R@1'])
  add('Sounds', M.sounds?.loc?.zero_shot_clip_classification, M.sounds?.cloud?.zero_shot_clip_classification)
  add('PDF pages', M.pdf?.loc?.page_image?.['R@1'], M.pdf?.cloud?.page_image?.['R@1'])
  add('Video', X.video?.loc?.['R@1'], X.video?.cloud?.['R@1'])
  add('Clustering', X.clustering?.loc_768?.v_measure, X.clustering?.cloud_768?.v_measure)
  add('Dedupe AP', X.dedupe?.loc?.ap, X.dedupe?.cloud?.ap)
  if (X.fleurs) {
    const v = Object.values(X.fleurs).filter((x: any) => x?.loc) as any[]
    if (v.length) add('Real voices', v.reduce((s, x) => s + x.loc['speech_to_own_transcript_R@1'], 0) / v.length, v.reduce((s, x) => s + x.cloud['speech_to_own_transcript_R@1'], 0) / v.length)
  }
  return { load, jobs, logs, rows, tests: rows.length }
}

const bar = (v: number, w: number) => '█'.repeat(Math.max(0, Math.round(v * w))).padEnd(w, '·')

export const register: Register = on => {
  let home = ''
  let last: Snapshot | undefined

  on('session.start', async ($, e, next) => {
    home = (await $.env.get('HOME')) ?? ''
    await $.command.register({ name: 'bench', description: 'Open the embedding benchmark live pane' })
    void $.ui.open({ id: PANE, title: 'Embedding bench' })
    $.clock.every(3000, async () => {
      last = await snapshot($, home).catch(() => last)
      if (last) $.ui.status(`bench · ${last.jobs.filter((x) => x.includes('suite') || x.includes('ann')).length} running · ${last.tests} tests · load ${last.load}`)
      $.ui.invalidate('ui.render')
    })
    return next(e)
  })

  on('command.run', { command: 'bench' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Embedding bench' })
    return { text: 'Benchmark pane opened.' }
  })

  on('tool.call', async ($, e, next) => {
    const input: any = e
    const detail = String(input.description ?? input.command ?? input.file_path ?? input.prompt ?? '').split('\n')[0].slice(0, 80)
    const one: Activity = { id: e.tool_use_id, tool: e.tool, detail, isDone: false }
    await update($, activity, list => [...list, one].slice(-50)).catch(() => undefined)
    const ran = await next(e)
    await update($, activity, list => list.map(a => (a.id === one.id ? { ...a, isDone: true } : a))).catch(() => undefined)
    return ran
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text } = $.ui.resolve(e)
    const s = last ?? (home ? await snapshot($, home) : undefined)
    const acts = await read($, activity)
    const w = Math.max(6, Math.min(16, Math.floor(((e.viewport?.columns ?? 80) - 40) / 2)))
    const feed = acts.slice(-6).map(a => <Text dimColor={a.isDone} wrap="truncate-end">{a.isDone ? '✓' : '▶'} {a.tool} {a.detail}</Text>)
    if (!s) return <Box flexDirection="column"><Text dimColor>Loading benchmark state…</Text><Text bold>What Claude is doing</Text>{feed}</Box>

    return (
      <Box flexDirection="column">
        <Text bold>Load {s.load} · {s.tests} tests scored · refreshes every 3 s</Text>
        <Text> </Text>
        <Text bold>Running</Text>
        {s.jobs.length === 0 && <Text dimColor>nothing running</Text>}
        {s.jobs.map(j => <Text color="yellow" wrap="truncate-end">{j}</Text>)}
        <Text> </Text>
        <Text bold>Latest log lines</Text>
        {s.logs.map(l => <Text wrap="truncate-end"><Text dimColor>{l.name}: </Text>{l.line}</Text>)}
        <Text> </Text>
        <Text bold>Scoreboard <Text color="green">local</Text> vs <Text color="blue">cloud</Text></Text>
        {s.rows.map(r => (
          <Text wrap="truncate-end">
            {r.label.padEnd(15)}<Text color="green">{bar(r.loc, w)}</Text> {r.loc.toFixed(3)}  <Text color="blue">{bar(r.cld, w)}</Text> {r.cld.toFixed(3)} {Math.abs(r.cld - r.loc) < 0.03 ? <Text dimColor>tie</Text> : r.loc > r.cld ? <Text color="green">local</Text> : <Text color="blue">cloud</Text>}
          </Text>
        ))}
        <Text> </Text>
        <Text bold>What Claude is doing</Text>
        {feed}
      </Box>
    )
  })
}
