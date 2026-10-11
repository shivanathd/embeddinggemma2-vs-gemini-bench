import { expect, test } from 'claude-code/testing'

test('shows each tool call in the pane activity feed', async ($, on) => {
  on('tool.call', async () => ({ result: { text: 'ok' } }) as never)
  await $.tool.call({ tool: 'Read', file_path: 'a.md' } as never)
  const ui = await $.ui.mount({ plugin: 'bench-watch', surface: 'terminal', component: 'Pane', requestId: 'bench-watch', viewport: { columns: 100, rows: 40 }, props: { title: 'Embedding bench', isFocused: false, bodyColumns: 100, placement: 'dock' } } as never)
  expect(await ui.find({ text: /Read/ })).toBeTruthy()
})
