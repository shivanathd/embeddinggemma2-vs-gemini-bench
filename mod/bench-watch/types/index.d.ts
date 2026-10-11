export type Activity = { id: string; tool: string; detail: string; isDone: boolean }

declare module 'claude-code' {
  interface PluginState {
    'bench-watch': { activity: Activity[] }
  }
}
