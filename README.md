（this readme is written by human）

# VCC: View-oriented Conversation Compiler

[English](README.md) | [简体中文](README_cn.md) | [日本語](README_jp.md)

Official implementation of "View-oriented Conversation Compiler for Agent Trace Analysis" ([Paper](https://arxiv.org/abs/2603.29678))

This repo is for daily use. To reproduce academic experiments in the paper, see [VCC-experiments](https://github.com/lllyasviel/VCC-experiments).

VCC is a compiler that compiles your conversation logs (Claude Code's JSONL) into efficient and agent-friendly views. Then you will never fear Claude Code's `/compact` - CC now can finally see the original details of compacted context. It also supports searching across all your Claude Code conversations.

Things that tend to happen after you install VCC:

- You find out that if you installed this earlier, you can save lots of money that you would need to waste for struggling with CC's `/compact`.
- `/compact` + `/recall` becomes your favorite combo.
- You start wondering why this wasn't built in CC.
- You delete your multi-layer RAG memory system, your self-evolving agent skill memory, and 15 other AGI inventions - if you really have those...

There is also a [paper](https://arxiv.org/abs/2603.29678) showing that VCC improves context learning and agent trace analysis and harness on some academic benchmark.

# Install

Currently We support Claude Code only. Codex and OpenClaw are coming soon.

To install, copy this to your Claude Code:

    Please help me install the skills from 
    https://github.com/lllyasviel/VCC.git 
    just clone it then follow the INSTALL.md

To update, copy this to your Claude Code:

    Please help me update the skills from
    https://github.com/lllyasviel/VCC.git
    just clone it then follow the INSTALL.md

To uninstall, copy this to your Claude Code:

    Please help me uninstall VCC skills by deleting
    `conversation-compiler`, `readchat`, `recall`, `searchchat`
    from my `.claude/skills`

Restart Claude Code after install, update, or uninstall.

# Basic Usage

When a compact is triggered (whether automaticaly or manually), you can immediately `/recall`.

For example, after an automatic compact or manual `/compact`, you will see somthing like `(... compacted)`. You can then:

`/recall`

and CC will recall everything automatically, or you can

`/recall let's review our conversation`

`/recall in those six rounds of analysis we just did, which access attempt causes the second layer of the server's three-layer log to crash?`

`/recall go through all six details I mentioned about the user survey again`

`/recall review my thought process so far`

For new conversations, you can use `/searchchat` or `/recall` (which will fallback to search). For example:

`/searchchat how did we handle the captcha in that browser listener system we discussed last time?`

`/recall did we decide to use React for that canvas solution we discussed last time?`

These commands can find and recover the orginal chats accross months of conversation history.

`/readchat` is for advanced usage, see below.

# How It Works

A Claude Code JSONL like this (you can find lots of these things in your `~/.claude/projects`):

```
{"type":"user","message":{"id":"msg_user1","content":"I have two pets.\nCan you write a P..."}}
{"type":"assistant","message":{"id":"msg_asst1","content":[{"type":"thinking","thinking":"The
user wants a pet tracking module.\nThey have a dog (Buddy) and a cat (Whiskers).\nLet me chec
k if there's an existing file first...
```

gets compiled into these views:

## UI View

This simulates what the user see in CC

```
[user]

I have two pets.
Can you write a Python module for tracking them?
One is a dog named Buddy and a cat named Whiskers.

[assistant]

Sure! Let me check if there's an existing file.

* Read "src/pets.py" (example.txt:18-20,23-25)

No existing file.
I'll create a module with Dog and Cat classes.

* Write "src/pets.py" (example.txt:40-63,66-68)

Created src/pets.py.
```

The pointers like `(example.txt:18-20,23-25)` can jump to the exact lines for full detail as below.

## Full View

The complete transcipt. Line numbers are assigned once and shared accross all views.

Lets say this is `example.txt`:

```
  1  [user]
  2
  3  I have two pets.
  4  Can you write a Python module for tracking them?
  5  One is a dog named Buddy and a cat named Whiskers.
  6
  7  ══════════════════════════════
  8  [assistant]
  9
 10  >>>thinking
 11  The user wants a pet tracking module.
 12  They have a dog (Buddy) and a cat (Whiskers).
 13  Let me check if there's an existing file first.
 14  <<<thinking
 15
 16  Sure! Let me check if there's an existing file.
 17
 18  >>>tool_call Read:tu01
 19  file_path: src/pets.py
 20  <<<tool_call
 21
 22  ══════════════════════════════
 23  [tool_error] Read:tu01
 24
 25  Error: file not found src/pets.py
 26
 27  ══════════════════════════════
 28  [assistant]
 29
 30  >>>thinking
 31  No existing file.
 32  I'll create Dog and Cat subclasses.
 33  The dog needs walk tracking.
 34  The cat needs nap tracking.
 35  <<<thinking
 36
 37  No existing file.
 38  I'll create a module with Dog and Cat classes.
 39
 40  >>>tool_call Write:tu02
 41  file_path: src/pets.py
 42  content: |
 43    class Pet:
 44        def __init__(self, name, species):
 45            self.name = name
 46            self.species = species
 47
 48    class Dog(Pet):
 49        def __init__(self, name):
 50            super().__init__(name, 'dog')
 51            self.walks = []
 52
 53        def log_walk(self, duration, distance):
 54            self.walks.append({'min': duration, 'km': distance})
 55
 56    class Cat(Pet):
 57        def __init__(self, name):
 58            super().__init__(name, 'cat')
 59            self.naps = []
 60
 61        def log_nap(self, duration, spot):
 62            self.naps.append({'min': duration, 'spot': spot})
 63  <<<tool_call
 64
 65  ══════════════════════════════
 66  [tool] Write:tu02
 67
 68  File written successfully.
 69
 70  ══════════════════════════════
 71  [assistant]
 72
 73  Created src/pets.py.
```

## Adaptive View `--grep dog`

The view can be adapted to search queries like `--grep dog`. Only blocks contaning a match are shown, but the conversation struture is preserved. Each block has a pointer back to the full view.

```
[user]

(example.txt:3-5)
  5: One is a dog named Buddy and a cat named Whiskers.

══════════════════════════════
[assistant]

>>>thinking
(example.txt:11-13)
  12: They have a dog (Buddy) and a cat (Whiskers).
<<<thinking

══════════════════════════════
[assistant]

>>>thinking
(example.txt:31-34)
  33: The dog needs walk tracking.
<<<thinking

>>>tool_call Write:tu02
(example.txt:41-62)
  50:           super().__init__(name, 'dog')
<<<tool_call
```

`Write:tu02` spans 22 lines of code (41–62), but only line 50 (`'dog'`) matched: the Cat class (56–62) is absent. The pointer `41-62` tells the agent where to read the full block.

## Transposed View `--grep dog`

Same matches, but as a flat list. Each entry is taged with what it is (user message, thinking, tool call, etc.) and where to find it in the full view:

```
(example.txt:3-5) [user]
  5: One is a dog named Buddy and a cat named Whiskers.

(example.txt:11-13) [thinking]
  12: They have a dog (Buddy) and a cat (Whiskers).

(example.txt:31-34) [thinking]
  33: The dog needs walk tracking.

(example.txt:41-62) [tool_call]
  50:           super().__init__(name, 'dog')
```

The adaptive view keeps conversation order so it is good for understanding context around a match. The transposed view is a flat list so it is good for scanning all matches at once. All pointers points into the full view.

## Additions in this fork

* **Timestamps**: `[user]` headers and every `--grep` hit show the UTC time (`[user] 2026-04-02 10:31Z`).
* **Search filters**: `--role tool_error`, `--since 2026-04-01`, `--until 2026-04-30`, `--limit 30`. They apply to both the stdout hits and `.view.txt`. `--grep . --role tool_error` lists every failed tool call.
* **Files view** (`.files.txt`): every file the conversation read or modified, newest first, with `(call, result)` pointers into the full view. `/recall` uses it as the checklist of files to re-read.
* **BM25 ranking** (`--rank`): the regex selects blocks, BM25 orders them by relevance to the pattern's words (rarer terms and shorter, focused blocks first). Use an alternation of synonyms as the query: `--grep "captcha|recaptcha|bot.?detect" --rank --limit 20`. Works with CJK text.
* **Graph view** (`--graph` → `.graph.txt`): each user turn linked to the files it read/modified, its tool errors, and `[prev Tn]` links to the last earlier turn on the same file. With several JSONL files, stdout gets a cross-conversation file index ("where else did we work on this file?").
* **Optional hook** (`vcc_hook.py`): compiles views when compaction starts, and/or archives logs so they survive Claude Code's 30-day `cleanupPeriodDays` deletion. See INSTALL.md.
* **Faster**: lowering no longer rescans the whole IR per section (~1.8x faster on large logs, identical output).
* **Tests**: `python -m unittest discover tests`.

All of it is deterministic and computed from the JSONL (no LLM calls, no stored summaries). Line numbers are unchanged: timestamps are appended to existing header lines.

# Q&A

### "Just another agent memory system?"

Nope. Memory systems store precomputed stuff like summaries, embeddings, graphs. Those structures and levels are usually static. And most of them are calling LLMs to do summarization etc...

VCC stores nothing. Views are dynamic. They are computed on the fly from the original JSONL, then thrown away after use. (We call this "projection".)

### "Okay isn't this just grep?"

No not at all. Grep gives you matching lines, but you cant tell if a match is the user talking, the agent thinking, a tool call, or a tool result. VCC has **block range pointers** and **block roles**. 

Some people may keep asking *"Okay, then what about splitting chat log into message files and using filesystem grep? What about structured grep? What about my XXXX database system?"* 

Lets say you jump from a adaptive view to the full view with some block line number, the surrounding context is right there. To get that from a file-per-message split, you need a tree to track hierarchy as a tree-like thing and then use a linked list to track temporal order, and you maintain both, and you even need to determine how many "temporal surrounding" things are really the correct surrounding context. By the time you finally finally make it work you basically reimplemented VCC..

### "Well but isn't this just pretty-print?"

They are very different. Pretty-print just reformats text. VCC is a real compiler with lex, parse, IR, lower, emit. Some examples of what it does:

* The lexer drops junk records like `queue-operation`, `progress`, `api_error` before parsing even starts
* The parser turns tool call parameters from escaped JSON blobs into readable block-indented text
* The parser also strips `digits→` prefixes from Read tool results to recover original source, and decodes base64 images to files
* At the IR stage, split assistant messages (same ID but multiple JSONL records because of compaction) get reassembled into single sections
* Lowering strips harness XML (`<system-reminder>`, `<ide_opened_file>`, etc), filters internal tools (`TodoWrite`, `ToolSearch`), cleans ANSI escape codes, and hides pure-markup user turns
* The emitter produces three views sharing one line-number coordinate system

the IR stage assign line numbers once to ensure that everything si consistent. After that, lowering can only select, truncate, or annotate. The line numbers cannot reorder or renumber. So cross-view pointers are always consistent.

# Cite

I haven't submitted VCC to any conferences or journals; just want it to be publicly available:

    @article{zhang2026vcc,
      title={View-oriented Conversation Compiler for Agent Trace Analysis},
      author={Lvmin Zhang and Maneesh Agrawala},
      year={2026},
      url={https://github.com/lllyasviel/VCC}
    }
