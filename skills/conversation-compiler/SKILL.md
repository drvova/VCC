---
name: conversation-compiler
description: "VCC (View-oriented Conversation Compiler) documentation. Compile coding-agent conversation logs (Claude Code, Codex CLI, Gemini CLI, or generic chat JSONL) into adaptive views."
---

# VCC — View-oriented Conversation Compiler

Compile coding-agent conversation logs into adaptive views for reading, searching, and context recovery.

## Usage

```bash
python "path/to/VCC.py" <input.jsonl ...> [options]
```

| Option | Description |
|--------|-------------|
| `-o <dir>` | Output directory (default: same as input) |
| `-t <N>` | Token truncation limit (default 128) |
| `-tu <N>` | User message token limit (default 256) |
| `--grep <pattern>` | Regex search pattern (Python `re` — use `a|b`, NOT `a\|b`) |
| `--role <types>` | With `--grep`: only these block types, comma-separated: `user`, `assistant`, `thinking`, `tool_call`, `tool_result`, `tool_error`, `system`. A prefix matches all subtypes (`tool` = all tool blocks) |
| `--since <when>` / `--until <when>` | With `--grep`: only blocks in this UTC time range. `YYYY-MM-DD` or ISO-8601; a bare `--until` date includes that day |
| `--limit <N>` | With `--grep`: print at most N matching blocks, newest first, and report how many older ones were left out |
| `--rank` | With `--grep`: order hits by BM25 relevance to the words in the pattern instead of newest first; each hit shows `bm25=<score>` |
| `--graph` | Also write `.graph.txt`; with several input files, print a cross-conversation file index to stdout |
| `--harness <name>` | Log format: `auto` (default, detected per file), `claude-code`, `codex`, `gemini-cli`, `chat` |

This tool also supports multi-file processing:

```bash
cd "/path/to/target/folder" && python "path/to/VCC.py" *.jsonl --grep "keyword"
```

## Supported logs

The format is detected per file (override with `--harness`). Harness-specific knowledge lives only in small input adapters and profiles; tool calls are understood from the shape of their arguments, not from tool names, so files/graph views work the same for every harness.

| Harness | Where the logs are | Notes |
|---|---|---|
| Claude Code | `~/.claude/projects/<project>/*.jsonl` (subagents: `<session>/subagents/*.jsonl`) | Deleted after 30 days by default (`cleanupPeriodDays`) |
| Codex CLI | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (`$CODEX_HOME` if set) | Files older than 7 days become `.jsonl.zst`; reading those needs Python 3.14+ or `pip install zstandard` |
| Gemini CLI | `~/.gemini/tmp/<project>/chats/session-*.jsonl` (subagents: `chats/<parent-id>/*.jsonl`; older: `session-*.json`) | |
| Anything else | Export to JSONL of `{"role", "content", "tool_calls"?, "timestamp"?}` messages (OpenAI/Anthropic style) | `--harness chat` |

Use globs that match the harness's file names, e.g. `rollout-*.jsonl*` for Codex, `session-*.json*` for Gemini CLI.

## Output Files

| File | When | Description |
|------|------|-------------|
| `.txt` | Always | Full transcript, lossless |
| `.min.txt` | Always | Brief overview |
| `.files.txt` | If any file tool ran | Files touched by Read/Write/Edit/MultiEdit/NotebookEdit, newest first, `M` = modified / `R` = read only, with line refs |
| `.graph.txt` | `--graph` only | One entry per user turn: its message, line range, tools used, errors, files read/modified, and `[prev Tn]` links to the last earlier turn that touched the same file; then a file → turns list |
| `.view.txt` | `--grep` only | Search-focused view |
| stdout | `--grep` only | Search results with block-level line range references |

## Rules

- **Forward slashes only in bash commands.** Backslashes in double quotes are escape characters — `\U`, `\l`, `\.` get silently mangled. Use `"C:/Users/..."` not `"C:\Users\..."`. This applies to ALL bash commands (not just this skill), on ALL platforms (Windows and Linux alike).
- Do NOT use `-o` unless the user explicitly requests an output directory. By default, compiled files are written next to the input JSONL — this is the intended behavior.
- Do NOT clean up compiled output files after use. Leave them in place unless the user explicitly asks to clean up.

## Workflow

**1. Compile**

```bash
python "path/to/VCC.py" "path/to/conversation.jsonl"
```

Produces `.txt` + `.min.txt` next to the input file. Long conversations are automatically split into numbered chunks. The console output lists every produced file with line/word counts — read it to understand the conversation's size and structure before proceeding:

```
conversation_1.txt  (2808 lines, 34545 words)
conversation_2.txt  (9622 lines, 116015 words)
conversation_3.txt  (3242 lines, 37946 words)
conversation_1.min.txt  (101 lines, 1572 words)
conversation_2.min.txt  (811 lines, 9504 words)
conversation_3.min.txt  (205 lines, 3340 words)
```

Chunks are numbered chronologically — higher numbers are more recent. (During `/recall`, the last chunk is the current conversation; the second-to-last is likely where the previous session's context ends.) Use this to gauge how many chunks exist and how large each is, so you can target your `.min.txt` reads and `--grep` searches to the right chunks.

**2. Read `.min.txt`**

The `.min.txt` is a scannable outline: user/assistant text is kept (truncated), tool calls collapse to one-line summaries with line references back to `.txt`.

```
[user]

Take a look!

[assistant]

Let me take a look.

* Read "code.py" (a_1.txt:19-21,24-34)
```

Each `*` line shows the tool name, key parameter, and two `.txt` line ranges: the call (19-21) and the result (24-34). Read these ranges in `.txt` for full details.

`[user]` headers carry the UTC time of the message (e.g. `[user] 2026-04-02 10:31Z`), so you can tell when things were said.

**3. Search with `--grep`**

Always use `--grep`, never system grep nor your embedded Grep tool. This script's `--grep` returns important block-level line RANGES that no other grep tools can provide. The output paths are relative to CWD, so `cd` close to the target first to keep outputs short and save tokens.

```bash
cd "/path/to/target/folder" && python "path/to/VCC.py" "path/to/conversation.jsonl" --grep "keyword"
```

Stdout example (`#` prefix = shortened filename)：

```
(#752a87.txt:L67-L81) [assistant] 2026-04-02 10:31Z   →  .txt lines 67-81, with block time
  77: one matched line ...                            →  .txt line 77
```

Optionally read `.view.txt` for focused search view.

Narrow searches instead of reading floods of hits:

- Time questions ("last week", "in March"): add `--since` / `--until`.
- Who said it: `--role user` for the user's own words, `--role assistant,thinking` for the agent's.
- What went wrong: `--grep . --role tool_error` lists every failed tool call.
- Many results: `--limit 30` shows the newest 30 and tells you how many older ones exist.
- Unsure of the exact wording: put synonyms and spelling variants in one alternation and rank them, e.g. `--grep "captcha|recaptcha|bot.?detect|anti.?bot" --rank --limit 20`. The regex picks which blocks match; `--rank` puts blocks that mention more (and rarer) terms, in less surrounding text, first.

**4. Jump to `.txt`**

All line references point to `.txt`. Read the referenced range for full context. Remember to read a bit more lines before and after the referenced range to get more context.

**5. Graph (optional)**

Add `--graph` to get `.graph.txt`, a map of the conversation by user turn:

```
T3 2026-04-03 15:30Z (a.txt:52-83) "Rename the dog to Rex"
  tools: Edit, Read
  errors: Bash (a.txt:70-74)
  M src/pets.py: Edit  [prev T1]
  subagent a04a577a (result a.txt:76) -> a/subagents/agent-a04a577a.jsonl
```

A `subagent` line means the turn started a subagent; when its log is found (searched by agent id under the log's directory) the path is shown: compile that file to read what the subagent did.

Use it to answer "which request changed this file, and what earlier turn touched it" (`[prev Tn]`), then jump to the turn's line range in `.txt`. With several JSONL files (`*.jsonl --graph`), stdout also lists, for each file, every turn in every conversation that touched it, newest first: use it to find "where else did we work on X".

# SKILLs

## /readchat — Read a specific conversation log

**Trigger**: The user wants to read a specific JSONL conversation log.

**Action**: Follow the Workflow above with the user-specified filenames or requests.

## /searchchat — Search across conversation logs

**Trigger**: The user wants to search across past conversation logs (see "Supported logs" for where each harness keeps them).

**Action**:

1. Find the log directory for the harness you are running in (table in "Supported logs"); `ls` it to narrow the search area (for Claude Code: `ls ~/.claude/projects/`).
2. `cd <log dir> && python "absolute/path/to/VCC.py" *.jsonl --grep "keyword"` — search top-level conversations first (Codex: `cd ~/.codex/sessions && ... "**/rollout-*.jsonl*"`).
3. If no results: `cd <log dir> && python "absolute/path/to/VCC.py" **/*.jsonl --grep "keyword"` — expand to subagents.

Across many conversations, add `--rank --limit 30` (most relevant 30 hits first) to keep output small, and `--since` when the user gives a time frame (logs last modified before `--since` are skipped without compiling, which is much faster). To find every conversation that worked on a given file, run `*.jsonl --graph` and read the file's entry in the printed index. If nothing is found for older conversations, the harness may have deleted them (Claude Code and Gemini CLI default to 30 days): see "Keeping your history" in INSTALL.md.

**Critical**: Always `cd` into the target directory first, then use VCC globs — a single `cd && python VCC.py <glob> --grep` is the correct pattern. Without `cd`, grep output contains full absolute paths instead of short relative paths, wasting tokens. All content search on JSONL must use VCC's `--grep`, never system grep or the embedded Grep tool — VCC's `--grep` returns block-level line ranges with role tags that no other grep can provide.

## /recall — Recover context from a previous conversation

**Trigger**: The conversation opens with a context-continuation summary ("This session is being continued from a previous conversation..."), or the user says `/recall`.

**Action**:

1. **Find the log file** — In Claude Code, the continuation message ends with `read the full transcript at: <path>.jsonl`. Otherwise (another harness, or `/recall` invoked manually), the current session is usually the most recently modified log in the harness's log directory ("Supported logs"); use the `/searchchat` method to locate it before proceeding to steps 2–3.

2. **Follow VCC Workflow** — Follow the full Workflow above (compile → read `.min.txt` → `--grep` → jump to `.txt`) with the log file found in step 1. The summary is lossy — only the original log is authoritative.

3. **Verify against current state** — After recovering the conversation, cross-check with reality:
   - **REALLY read files** — Use `.files.txt` as the checklist: it lists every file the conversation read or modified (`M` first priority). Writes done through shell commands are included only when they could be inferred (marked `~`, e.g. `Bash~`); scripts that write files from inside an interpreter (e.g. `python - <<EOF … open(p,'w')`) are not, so also scan `.min.txt` for shell commands that may write files. Read all files referenced in the conversation with your Read tool. Pre-loaded content from system-reminders or prior conversation turns does NOT count — it may be truncated, stale, or lossy. You must issue a fresh Read call for each file, even if you believe you already have its content. 99.97% of the time, the user has externally modified nearly every file between sessions. If you skip reading and miss their external changes, your failure rate is effectively 100%.
   - **Key details** — Identify specific values, paths, configs, and logic mentioned in the conversation. Compare against the actual files to catch drift or errors.
   - **Understand the journey** — Trace the user's intent, decision sequence, and direction changes. Understand not just what was done, but how and why they got there. For long conversations, compile with `--graph` and read `.graph.txt` first: it lists every user request in order with the files it changed and the errors it hit.
