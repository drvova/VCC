# Installation Guide for Agents (Manual Install)

## Requirements

- Python 3.10+

## Install

Clone this repo to any temporary location. Copy the four folders inside `skills/` (`conversation-compiler`, `readchat`, `recall`, `searchchat`) into your project's `.claude/skills/`. Delete the cloned repo.

## Update

Clone this repo to any temporary location. Delete the four folders (`conversation-compiler`, `readchat`, `recall`, `searchchat`) from your `.claude/skills/`, then copy the new versions from the cloned `skills/` into `.claude/skills/`. Delete the cloned repo.

## Keeping your history (recommended)

VCC can only read logs that still exist. By default Claude Code **deletes logs older than 30 days** at startup (`cleanupPeriodDays`, default 30). To search further back, raise it in `~/.claude/settings.json`:

```json
{ "cleanupPeriodDays": 3650 }
```

Gemini CLI also cleans up old sessions (30 days by default; see its session-management settings). Codex keeps rollouts but compresses those older than 7 days to `.jsonl.zst`, which VCC reads with Python 3.14+ or `pip install zstandard`.

For any harness, `vcc_hook.py --sweep` mirrors a log directory into an archive (log files only, copied only when newer; it never deletes anything). Run it from cron, e.g. hourly:

```
0 * * * * python3 /path/to/conversation-compiler/scripts/vcc_hook.py --archive ~/agent-log-archive --sweep ~/.claude/projects --sweep ~/.codex/sessions --sweep ~/.gemini/tmp
```

Search the archive like any log directory. A harness that later compresses or converts a log (Codex `.jsonl` → `.jsonl.zst`, Gemini CLI `.json` → `.jsonl`) can leave both copies in the archive, so a conversation may show up twice.

## Optional hooks (Claude Code)

`scripts/vcc_hook.py` (inside `conversation-compiler`) can run as a Claude Code hook. It never blocks or fails Claude Code: errors go to stderr and it always exits 0.

- `--compile`: build the views when compaction starts, so they exist on disk right away (for example for other tools). Compilation is fast (about 1s for a 30 MB log) and `/recall` recompiles anyway to include newer messages, so this mainly helps with very large logs.
- `--archive DIR`: copy the session log (and subagent logs) to `DIR/<project>/`, so it survives cleanup. Run `/searchchat` against the archive directory to search it.

Add to `.claude/settings.json` (project) or `~/.claude/settings.json` (all projects); adjust the script path to where the skill is installed:

```json
{
  "hooks": {
    "PreCompact": [
      { "hooks": [ { "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/skills/conversation-compiler/scripts/vcc_hook.py\" --compile --archive ~/.claude/vcc-archive" } ] }
    ],
    "SessionEnd": [
      { "hooks": [ { "type": "command",
        "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/skills/conversation-compiler/scripts/vcc_hook.py\" --archive ~/.claude/vcc-archive" } ] }
    ]
  }
}
```

Only add these hooks if the user asks for them.

## Verify

After install or update, ask the user to restart Claude Code, then run `/recall` to test that everything works.

## Uninstall

Delete `conversation-compiler`, `readchat`, `recall`, `searchchat` from your `.claude/skills/`. Ask the user to restart Claude Code.
