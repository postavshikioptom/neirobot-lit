# Codex Agents Migration

В этой папке перенесены Claude Code CLI агенты в нативный для Codex формат custom agents.

Что лежит здесь:

- `.codex/agents/*.toml` — Codex custom agents с теми же именами: `coder`, `lit-expert`, `planner`, `searcher`, `train-expert`
- `.codex/agent-memory/*` — перенесенная память агентов из `.claude/agent-memory`

Как это использовать в Codex:

- Codex читает project `AGENTS.md` отдельно, а custom agents подхватывает из `.codex/agents/`
- Эти агенты являются аналогом Claude subagents: их можно вызывать по имени при делегировании/спавне подагента
- Инструкции перенесены из `.claude/agents/*.md` в поля `name`, `description`, `developer_instructions`

Примечания:

- Пути памяти внутри инструкций переведены с `.claude/agent-memory` на `.codex/agent-memory`
- Для совместимости с исходной инструкцией `train-expert` создана также папка `.codex/agent-memory/train_expert`