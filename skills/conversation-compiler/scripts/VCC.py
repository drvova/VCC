#!/usr/bin/env python3
"""View-oriented Conversation Compiler (VCC) - compile Claude JSONL logs into adaptive views.

Produces per conversation chain:
  .txt       Full transcript (lossless rendering of JSONL)
  .min.txt   Brief mode (tool_call summaries with line refs, no separators)
  .files.txt Files view (files touched by Read/Write/Edit, with line refs; if any)
  .graph.txt Graph view (user turns <-> files touched, errors, links; only with --graph)
  .view.txt  View mode (search-focused, only with --grep)

Usage:
  python VCC.py conversation.jsonl              # .txt + .min.txt
  python VCC.py conversation.jsonl --grep "kw"  # + .view.txt + stdout search hits
  python VCC.py conversation.jsonl -t 128       # truncation limit (tokens, default 128)
  python VCC.py conversation.jsonl -tu 256      # user message truncation limit (default 256)
  python VCC.py conversation.jsonl -o outdir    # output directory
  python VCC.py project/*.jsonl --grep "kw"     # multi-file search
  python VCC.py *.jsonl --grep "kw" --since 2026-04-01 --role user,assistant --limit 20
  python VCC.py s.jsonl --grep . --role tool_error  # every failed tool call
  python VCC.py *.jsonl --grep "captcha|recaptcha|bot.?detect" --rank --limit 20  # BM25 order
  python VCC.py *.jsonl --graph                     # + cross-conversation file index on stdout
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import glob as globmod

# ── dict emitter ──

def _emit_dict(data, indent=0):
    parts = []
    prefix = " " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            parts.append(prefix + k + ":")
            parts.append(_emit_dict(v, indent + 2))
        elif isinstance(v, list):
            parts.append(prefix + k + ":")
            for item in v:
                if isinstance(item, dict):
                    parts.append(prefix + "  -")
                    parts.append(_emit_dict(item, indent + 4))
                else:
                    parts.append(prefix + "  - " + str(item))
        else:
            s = str(v)
            if "\n" in s:
                parts.append(prefix + k + ": |")
                p = " " * (indent + 2)
                for line in s.split("\n"):
                    parts.append((p + line) if line else "")
            else:
                parts.append(prefix + k + ": " + s)
    return "\n".join(parts)

# ── tokenizer ──

_TOK_RE = re.compile(
    r'[a-zA-Z]+'           # letters (grouped)
    r'|[0-9]+'             # digits (grouped)
    r'|[^\sa-zA-Z0-9]'     # single char: any non-whitespace non-letter non-digit
    r'|\s+'                # whitespace (preserved, not counted)
)
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

def _tokenize(text):
    return _TOK_RE.findall(text)

# ── truncation (token-based) ──

def _trunc(text, limit, ref=""):
    if not limit or not text:
        return text
    tokens = _tokenize(text)
    count, cut = 0, len(tokens)
    for i, t in enumerate(tokens):
        if t.strip():
            count += 1
            if count > limit:
                cut = i
                break
    if cut >= len(tokens):
        return text
    return "".join(tokens[:cut]) + (f"...(truncated from {ref})" if ref else "...(truncated)")

# ── match lines ──

def match_lines(lines, regex, ref_fn="x.txt", start_line=1):
    if not lines:
        return []
    end_line = start_line + len(lines) - 1
    from_ref = f"...(from {ref_fn}:{start_line}-{end_line})"

    matched = []
    for i, line in enumerate(lines):
        if regex.search(line):
            matched.append((start_line + i, line))
    if not matched:
        return [from_ref]

    block_ref = f"({ref_fn}:{start_line}-{end_line})"
    result = [block_ref]
    for ln, lt in matched:
        result.append(f"  {ln}: {lt}")
    return result


# ── lexer ──

SEP = "══════════════════════════════"
_DISCARD_T = {"queue-operation", "file-history-snapshot", "last-prompt", "progress"}
_DISCARD_S = {"stop_hook_summary", "api_error", "bridge_status", "informational", "local_command"}

def _short(fn):
    n, e = os.path.splitext(fn)
    return ("#" + n[-6:] + e) if len(n) > 12 else fn

def _short_tid(tid):
    return tid[-6:] if len(tid) > 6 else tid

# ── timestamps ──

def _parse_ts(s):
    """Parse an ISO-8601 timestamp (JSONL style, e.g. 2026-04-02T10:31:05.123Z) to aware UTC."""
    from datetime import datetime, timezone
    if not s or not isinstance(s, str):
        return None
    try:
        d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)

def _fmt_ts(s):
    d = _parse_ts(s)
    return d.strftime("%Y-%m-%d %H:%MZ") if d else None

def _parse_when(s, end=False):
    """Parse a --since/--until value. A bare date (YYYY-MM-DD) as --until covers that whole day."""
    from datetime import timedelta
    d = _parse_ts(s)
    if d is None:
        raise ValueError(f"cannot parse time {s!r} (use YYYY-MM-DD or ISO-8601)")
    if end and len(s.strip()) == 10:
        d += timedelta(days=1)
    return d

def lex(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _collect_stats(chain):
    """Extract usage/timing/model stats from a chain of records, including subagents."""
    from collections import defaultdict
    from datetime import datetime
    totals = defaultdict(int)
    models = set()
    timestamps = []
    api_calls = 0
    tool_uses = 0
    subagent_tokens = 0
    for r in chain:
        msg = r.get("message", {})
        ts = r.get("timestamp")
        if ts:
            timestamps.append(ts)
        usage = msg.get("usage")
        if usage:
            api_calls += 1
            for k, v in usage.items():
                if isinstance(v, int):
                    totals[k] += v
        model = msg.get("model")
        if model:
            models.add(model)
        content = msg.get("content", [])
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tool_uses += 1
        # subagent usage from toolUseResult
        tur = r.get("toolUseResult")
        if isinstance(tur, dict) and tur.get("agentId"):
            try:
                subagent_tokens += int(tur.get("totalTokens", 0))
            except (ValueError, TypeError):
                pass
    duration = None
    if len(timestamps) >= 2:
        try:
            t0 = datetime.fromisoformat(min(timestamps).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(max(timestamps).replace("Z", "+00:00"))
            duration = int((t1 - t0).total_seconds())
        except Exception:
            pass
    if api_calls == 0:
        return None
    lines = [SEP, "[stats]", ""]
    lines.append(f"model: {', '.join(sorted(models))}")
    lines.append(f"api_calls: {api_calls}  tool_uses: {tool_uses}")
    if duration is not None:
        m, s = divmod(duration, 60)
        lines.append(f"duration: {m}m{s:02d}s" if m else f"duration: {s}s")
    inp = totals.get("input_tokens", 0)
    cr = totals.get("cache_read_input_tokens", 0)
    cc = totals.get("cache_creation_input_tokens", 0)
    out = totals.get("output_tokens", 0)
    own = inp + cr + cc + out
    own_eff = int(inp * 1.0 + cr * 0.1 + cc * 1.25 + out * 5.0)

    def _fmt_block(label, raw, eff):
        return f"{label}: {raw:,} (effective: {eff:,})"

    if subagent_tokens:
        lines.append("")
        lines.append("Subagents:")
        lines.append(f"  total: {subagent_tokens:,}")
        lines.append("")
        lines.append("Main:")

    parts = []
    if inp: parts.append(f"input: {inp:,}")
    if cr: parts.append(f"cache_read: {cr:,}")
    if cc: parts.append(f"cache_create: {cc:,}")
    pfx = "  " if subagent_tokens else ""
    if parts:
        lines.append(f"{pfx}{'  '.join(parts)}")
    lines.append(f"{pfx}output: {out:,}")
    lines.append(f"{pfx}{_fmt_block('total', own, own_eff)}")

    if subagent_tokens:
        all_raw = own + subagent_tokens
        all_eff = own_eff + subagent_tokens
        lines.append("")
        lines.append("All:")
        lines.append(f"  {_fmt_block('total', all_raw, all_eff)}")
    return lines

def _sanitize(text):
    if not text:
        return text
    if "\r" in text:
        text = text.replace("\r", "")
    if "\x1b" in text:
        text = _ANSI_RE.sub("", text)
    return _CTRL_RE.sub("", text)

def _preprocess_tool_text(text, tool_name):
    text = _sanitize(text)
    if tool_name != "Read":
        return text
    lines = []
    for line in text.split("\n"):
        if "→" in line:
            head, tail = line.split("→", 1)
            if head.strip().isdigit():
                line = tail
        lines.append(line)
    return "\n".join(lines)

def _discard(r):
    t = r.get("type")
    return t in _DISCARD_T or (t == "system" and r.get("subtype") in _DISCARD_S)

def merge_chunks(recs):
    merged = []
    active_mid = None
    active_idx = None
    for r in recs:
        if r.get("type") == "assistant":
            m = r.get("message", {})
            mid = m.get("id")
            if mid and mid == active_mid and active_idx is not None:
                merged[active_idx]["message"]["content"].extend(m.get("content", []))
                if m.get("stop_reason"):
                    merged[active_idx]["message"]["stop_reason"] = m["stop_reason"]
            else:
                merged.append(r)
                if mid:
                    active_mid = mid
                    active_idx = len(merged) - 1
                else:
                    active_mid = None
                    active_idx = None
        else:
            merged.append(r)
            if not _discard(r):
                active_mid = None
                active_idx = None
    return merged

def split_chains(recs):
    kept = [r for r in recs if not _discard(r)]
    chains, cur = [], []
    for r in kept:
        if r.get("type") == "system" and r.get("subtype") == "compact_boundary":
            if cur: chains.append(cur)
            cur = []
        else:
            cur.append(r)
    if cur: chains.append(cur)
    return chains

# ── image / doc ──

def _media_ext(media_type, default_ext):
    if "/" not in media_type:
        return default_ext
    ext = media_type.split("/", 1)[1].split("+", 1)[0]
    if ext == "jpeg":
        return "jpg"
    return ext or default_ext

def _extract_base64(source, outdir, data_prefix, data_index, stem, default_mt, default_ext):
    mt = source.get("media_type", default_mt)
    fn = f"{data_prefix}_{stem}_{data_index}.{_media_ext(mt, default_ext)}"
    with open(os.path.join(outdir, fn), "wb") as f:
        f.write(base64.b64decode(source.get("data", "")))
    return fn

def _extract_img(source, outdir, data_prefix, data_index):
    return _extract_base64(source, outdir, data_prefix, data_index,
                           "img", "image/png", "png")

def _extract_doc(source, outdir, data_prefix, data_index):
    return _extract_base64(source, outdir, data_prefix, data_index,
                           "doc", "application/octet-stream", "bin")


# ── tool_call summary ──

_TOOL_SUMMARY_FIELDS = {
    "Read": "file_path", "Edit": "file_path", "Write": "file_path",
    "Glob": "pattern", "Grep": "pattern",
    "Agent": "description", "Skill": "skill",
}
# Tools whose input names a file; used by the files view.
_FILE_TOOLS = {"Read", "Write", "Edit", "MultiEdit", "NotebookEdit"}
_FILE_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

def _tool_summary(name, inp):
    """Build one-line summary: * Name "param" """
    field = _TOOL_SUMMARY_FIELDS.get(name)
    if field and field in inp:
        return f'* {name} "{inp[field]}"'
    if name == "Bash":
        val = inp.get("description") or inp.get("command", "")
        if val:
            # truncate long commands
            if not inp.get("description") and len(val) > 60:
                val = val[:57] + "..."
            return f'* {name} "{val}"'
    return f"* {name}"


# ── IR node ──

def _node(typ, content, **kw):
    o = {"type": typ, "content": content,
         "searchable": kw.pop("searchable", False)}
    o.update(kw)
    return o

# ── parser ──

def parse(chain, outdir, data_prefix, data_ctr):
    ir = []
    sec = 0
    blk = 0

    tid_name = {}
    for r in chain:
        if r.get("type") == "assistant":
            for b in r.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    tid_name[b.get("id", "")] = b.get("name", "unknown")

    def _emit_sep():
        if sec > 0:
            ir.append(_node("meta", ["", SEP]))

    def _emit_header(h):
        ir.append(_node("meta_header", [h, ""], _sec=sec))

    def _emit_blocks(blocks, text_type):
        nonlocal blk
        has_any = False
        for b in blocks:
            bt = b.get("type")
            if bt == "thinking":
                txt = _sanitize(b.get("thinking", ""))
                if not txt: continue
                ir.append(_node("meta", [">>>thinking"], _sec=sec, _blk=blk))
                ir.append(_node("thinking", txt.split("\n"), searchable=True,
                                 _sec=sec, _blk=blk))
                ir.append(_node("meta", ["<<<thinking"], _sec=sec, _blk=blk))
                blk += 1; has_any = True

            elif bt == "redacted_thinking":
                ir.append(_node("meta", [">>>redacted_thinking"], _sec=sec, _blk=blk))
                ir.append(_node("redacted_thinking",
                                 ["[content redacted by model provider]"],
                                 searchable=True, _sec=sec, _blk=blk))
                ir.append(_node("meta", ["<<<redacted_thinking"], _sec=sec, _blk=blk))
                blk += 1; has_any = True

            elif bt == "text":
                txt = _sanitize(b.get("text", ""))
                if not txt: continue
                ir.append(_node(text_type, txt.split("\n"), searchable=True,
                                 _sec=sec, _blk=blk))
                blk += 1; has_any = True

            elif bt == "tool_use":
                name = b.get("name", "unknown")
                tid = b.get("id", "")
                inp = b.get("input", {})
                hl = f">>>tool_call {name}:{_short_tid(tid)}"
                summary = _tool_summary(name, inp)
                fpath = None
                if name in _FILE_TOOLS and isinstance(inp, dict):
                    fpath = inp.get("file_path") or inp.get("notebook_path")
                ir.append(_node("meta", [hl], _sec=sec, _blk=blk,
                                 _tool_summary=summary, _tool_name=name,
                                 _tool_tid=_short_tid(tid), _tool_file=fpath))
                if inp:
                    ir.append(_node("tool_call", _emit_dict(inp).split("\n"),
                                     searchable=True, _sec=sec, _blk=blk))
                ir.append(_node("meta", ["<<<tool_call"], _sec=sec, _blk=blk))
                blk += 1; has_any = True

            elif bt == "image":
                src = b.get("source", {})
                if src.get("type") == "base64":
                    fn = _extract_img(src, outdir, data_prefix, data_ctr[0])
                    data_ctr[0] += 1
                    ir.append(_node(f"{text_type}_image", [f"[image: {fn}]"],
                                     searchable=True, _sec=sec, _blk=blk))
                    blk += 1; has_any = True

            elif bt == "document":
                src = b.get("source", {})
                label = "[document]"
                if src.get("type") == "base64":
                    fn = _extract_doc(src, outdir, data_prefix, data_ctr[0])
                    data_ctr[0] += 1
                    label = f"[document: {fn}]"
                ir.append(_node(f"{text_type}_document", [label],
                                 searchable=True, _sec=sec, _blk=blk))
                blk += 1; has_any = True
        return has_any

    def _user_header(r):
        ts = _fmt_ts(r.get("timestamp"))
        return f"[user] {ts}" if ts else "[user]"

    # Every IR node is stamped with the timestamp of the record it came from
    # (used for --since/--until and for dating search hits).
    stamp_from, cur_ts = 0, None
    for r in chain:
        for o in ir[stamp_from:]:
            o.setdefault("_ts", cur_ts)
        stamp_from, cur_ts = len(ir), r.get("timestamp")
        rt = r.get("type")

        if rt == "system":
            if r.get("subtype") == "compact_boundary": continue
            content = r.get("content", "") or r.get("message", {}).get("content", "")
            if not content: continue
            _emit_sep(); _emit_header("[system]")
            if isinstance(content, list):
                _emit_blocks(content, "system")
            else:
                ir.append(_node("system", _sanitize(content).split("\n"), searchable=True,
                                 _sec=sec, _blk=blk))
                blk += 1
            sec += 1

        elif rt == "user":
            if r.get("isCompactSummary"):
                content = r.get("message", {}).get("content", "")
                nlines = content.count("\n") + 1 if content else 0
                _emit_sep(); _emit_header(_user_header(r))
                ir.append(_node("user", [f"[compact summary — {nlines} lines]"], searchable=False,
                                 _sec=sec, _blk=blk))
                blk += 1; sec += 1
                continue
            content = r.get("message", {}).get("content", "")
            if isinstance(content, str):
                if content:
                    _emit_sep(); _emit_header(_user_header(r))
                    ir.append(_node("user", _sanitize(content).split("\n"), searchable=True,
                                     _sec=sec, _blk=blk))
                    blk += 1; sec += 1
            elif isinstance(content, list):
                tblocks = [b for b in content if b.get("type") != "tool_result"]
                tresults = [b for b in content if b.get("type") == "tool_result"]
                if tblocks:
                    mark = len(ir)
                    saved_data = data_ctr[0]
                    saved_blk = blk
                    _emit_sep(); _emit_header(_user_header(r))
                    if _emit_blocks(tblocks, "user"):
                        sec += 1
                    else:
                        del ir[mark:]
                        data_ctr[0] = saved_data
                        blk = saved_blk
                for tr in tresults:
                    tuid = tr.get("tool_use_id", "")
                    is_err = tr.get("is_error", False)
                    nm = tid_name.get(tuid, "unknown")
                    role = "tool_error" if is_err else "tool"
                    btype = "tool_error" if is_err else "tool_result"
                    _emit_sep(); _emit_header(f"[{role}] {nm}:{_short_tid(tuid)}")
                    tc = tr.get("content", "")
                    parts = []
                    if isinstance(tc, str):
                        parts.append(_preprocess_tool_text(tc, nm))
                    elif isinstance(tc, list):
                        for item in tc:
                            if item.get("type") == "text":
                                parts.append(_preprocess_tool_text(item.get("text", ""), nm))
                            elif item.get("type") == "image":
                                src = item.get("source", {})
                                if src.get("type") == "base64":
                                    fn = _extract_img(src, outdir, data_prefix, data_ctr[0])
                                    data_ctr[0] += 1
                                    parts.append(f"[image: {fn}]")
                            elif item.get("type") == "document":
                                src = item.get("source", {})
                                if src.get("type") == "base64":
                                    fn = _extract_doc(src, outdir, data_prefix, data_ctr[0])
                                    data_ctr[0] += 1
                                    parts.append(f"[document: {fn}]")
                    ir.append(_node(btype, _sanitize("\n\n".join(parts)).split("\n"),
                                     searchable=True, _sec=sec, _blk=blk))
                    blk += 1; sec += 1

        elif rt == "assistant":
            blocks = r.get("message", {}).get("content", [])
            has = any(
                (b.get("type") == "thinking" and b.get("thinking")) or
                b.get("type") == "redacted_thinking" or
                (b.get("type") == "text" and b.get("text")) or
                b.get("type") == "tool_use" or
                b.get("type") == "image" or
                b.get("type") == "document"
                for b in blocks)
            if has:
                _emit_sep(); _emit_header("[assistant]")
                _emit_blocks(blocks, "assistant")
                sec += 1

    for o in ir[stamp_from:]:
        o.setdefault("_ts", cur_ts)
    ir.append(_node("meta", [""]))  # trailing newline
    return ir


# ── IR walk ──

def _is_tool_summary(o):
    return o.get("_tool_summary") is not None

def _walk(ir, key="content"):
    prev_blk = None
    prev_o = None
    for o in ir:
        c = o.get(key)
        if c is None:
            continue
        if not c:
            continue
        blk = o.get("_blk")
        if blk is not None and prev_blk is not None and blk != prev_blk:
            if key == "content":
                blank = True
            else:
                # non-content: suppress blank between consecutive tool summaries
                blank = not (_is_tool_summary(prev_o) and _is_tool_summary(o))
        else:
            blank = False
        yield o, c, blank
        if blk is not None:
            prev_blk = blk
            prev_o = o
        elif SEP in o.get("content", []):
            prev_blk = None
            prev_o = None


# ── line assignment ──

def assign_lines(ir):
    line = 0
    for o, c, blank in _walk(ir, "content"):
        if blank: line += 1
        o["start_line"] = line
        line += len(c)
        o["end_line"] = line - 1
    return line


# ── lowering helpers ──

def _is_truncatable(o):
    t = o["type"]
    if t in ("meta", "meta_header", "thinking", "redacted_thinking"):
        return False
    if t.endswith("_image") or t.endswith("_document"):
        return False
    return True

def _is_thinking(o):
    return o["type"] in ("thinking", "redacted_thinking")

def _sec_roles(ir):
    """Map sec -> role from meta_header content."""
    roles = {}
    for o in ir:
        if o["type"] == "meta_header":
            s = o.get("_sec")
            if s is None: continue
            h = o["content"][0]
            if h.startswith("[tool_error]"):
                roles[s] = "tool_error"
            elif h.startswith("[tool]"):
                roles[s] = "tool"
            elif h.startswith("[assistant]"):
                roles[s] = "assistant"
            elif h.startswith("[user]"):
                roles[s] = "user"
            elif h.startswith("[system]"):
                roles[s] = "system"
    return roles


# ── brief-mode filtering ──

_BRIEF_STRIP_RE = re.compile(
    r'<(ide_opened_file|ide_selection|system-reminder|command-message)>.*?</\1>\s*',
    re.DOTALL)
_BRIEF_UNWRAP_RE = re.compile(r'</?(?:command-name|command-args)>')
_META_USER_RE = re.compile(
    r'^\s*<(?:task-notification|local-command-caveat|local-command-stdout'
    r'|local-command-stderr)\b')
_SKILL_USER_RE = re.compile(r'^Base directory for this skill:')
_BRIEF_HIDE_TOOLS = {"TodoWrite", "ToolSearch"}
_BRIEF_HIDE_EXACT = {"Continue from where you left off.", "No response requested."}

def _strip_noise_xml(text):
    """Strip known noise XML from user text for brief mode."""
    text = _BRIEF_STRIP_RE.sub('', text)
    text = _BRIEF_UNWRAP_RE.sub('', text)
    return text.strip()

def _searchable_by_sec(ir):
    """Map sec -> its searchable nodes, in IR order (built once per lowering)."""
    idx = {}
    for o in ir:
        if o.get("searchable") and o.get("_sec") is not None:
            idx.setdefault(o["_sec"], []).append(o)
    return idx

def _user_hidden_in_brief(ir, sec, by_sec=None):
    """Check if a user section should be entirely hidden in brief mode."""
    if by_sec is None:
        by_sec = _searchable_by_sec(ir)
    blocks = by_sec.get(sec, [])
    if not blocks:
        return False
    for o in blocks:
        text = "\n".join(o["content"]).strip()
        if not text:
            continue
        if _META_USER_RE.match(text):
            continue
        if _SKILL_USER_RE.match(text):
            continue
        if not _strip_noise_xml(text):
            continue
        return False
    return True

def _section_hidden_exact(ir, sec, by_sec=None):
    """Check if all searchable content in a section is an exact-match hide string."""
    if by_sec is None:
        by_sec = _searchable_by_sec(ir)
    blocks = [o for o in by_sec.get(sec, [])
              if o["type"] not in ("thinking", "redacted_thinking")]
    if not blocks:
        return False
    for o in blocks:
        text = "\n".join(o["content"]).strip()
        if text not in _BRIEF_HIDE_EXACT:
            return False
    return True


# ── lowering: brief ──

def _tid_result_ranges(ir):
    """Map short_tid → (start_line, end_line) for tool_result sections."""
    # Find which sec corresponds to which short_tid
    tid_sec = {}
    for o in ir:
        if o["type"] == "meta_header":
            c = o["content"]
            if c and len(c) >= 1:
                h = c[0]
                # [tool] name:AABBCC or [tool_error] name:AABBCC
                if h.startswith("[tool]") or h.startswith("[tool_error]"):
                    parts = h.split(":")
                    if len(parts) >= 2:
                        tid_sec[parts[-1]] = o.get("_sec")
    # Collect line ranges per sec
    sec_range = {}
    for o in ir:
        s = o.get("_sec")
        if s is None:
            continue
        sl = o.get("start_line")
        el = o.get("end_line")
        if sl is not None and el is not None:
            if s not in sec_range:
                sec_range[s] = [sl, el]
            else:
                sec_range[s][0] = min(sec_range[s][0], sl)
                sec_range[s][1] = max(sec_range[s][1], el)
    # Build tid → range (only content nodes, skip the separator/header)
    result = {}
    for tid, sec in tid_sec.items():
        if sec in sec_range:
            result[tid] = tuple(sec_range[sec])
    return result


def lower_brief(ir, truncate, filename="", truncate_user=256):
    short = _short(filename)
    roles = _sec_roles(ir)
    tid_ranges = _tid_result_ranges(ir)

    # Sections visible in truncation: not tool/tool_error/system
    visible_secs = {sec for sec, role in roles.items()
                     if role not in ("tool", "tool_error", "system")}

    # Determine which assistant sections to merge:
    # An assistant section merges if the previous VISIBLE section is also assistant.
    all_secs = sorted(roles.keys())
    merge_secs = set()
    for i in range(1, len(all_secs)):
        cur_sec = all_secs[i]
        if not (cur_sec in visible_secs and
                roles.get(cur_sec) == "assistant"):
            continue
        # Find previous visible section
        for j in range(i - 1, -1, -1):
            if all_secs[j] in visible_secs:
                if roles.get(all_secs[j]) == "assistant":
                    merge_secs.add(cur_sec)
                break

    # Track which secs only have thinking content (no visible non-thinking blocks)
    sec_has_nonthink = set()
    for o in ir:
        s = o.get("_sec")
        if s is None: continue
        if o["type"] not in ("meta", "meta_header", "thinking", "redacted_thinking"):
            sec_has_nonthink.add(s)

    # Sections that are all-thinking should still be hidden in truncation
    # (they have no visible content after thinking is removed)
    by_sec = _searchable_by_sec(ir)
    for sec in list(visible_secs):
        if sec not in sec_has_nonthink:
            visible_secs.discard(sec)
        elif roles.get(sec) == "user" and _user_hidden_in_brief(ir, sec, by_sec):
            visible_secs.discard(sec)
        elif _section_hidden_exact(ir, sec, by_sec):
            visible_secs.discard(sec)

    # Recompute merge_secs after possible visibility changes
    all_secs_2 = sorted(roles.keys())
    merge_secs = set()
    for i in range(1, len(all_secs_2)):
        cur_sec = all_secs_2[i]
        if not (cur_sec in visible_secs and
                roles.get(cur_sec) == "assistant"):
            continue
        for j in range(i - 1, -1, -1):
            if all_secs_2[j] in visible_secs:
                if roles.get(all_secs_2[j]) == "assistant":
                    merge_secs.add(cur_sec)
                break

    for idx, o in enumerate(ir):
        s = o.get("_sec")

        # Separator: replace with blank line in brief mode
        if s is None and o["type"] == "meta" and SEP in o.get("content", []):
            ns = None
            for j in range(idx + 1, len(ir)):
                ns = ir[j].get("_sec")
                if ns is not None:
                    break
            if ns is None or ns not in visible_secs:
                o["content_brief"] = None
            elif ns in merge_secs:
                o["content_brief"] = None
            elif not any(sec in visible_secs for sec in range(ns)):
                o["content_brief"] = None
            else:
                o["content_brief"] = [""]
            continue

        # Section not visible → hide
        if s is not None and s not in visible_secs:
            o["content_brief"] = None
            continue

        # Merged assistant: hide separator and header
        if s in merge_secs and o["type"] == "meta_header":
            o["content_brief"] = None
            continue

        # Thinking / redacted_thinking → hide (including their >>> <<< metas)
        if _is_thinking(o):
            o["content_brief"] = None
            continue
        if o["type"] == "meta":
            c = o["content"]
            if c and (c[0].startswith(">>>thinking") or c[0].startswith("<<<thinking") or
                      c[0].startswith(">>>redacted_thinking") or c[0].startswith("<<<redacted_thinking")):
                o["content_brief"] = None
                continue

        # Tool_call three-piece: collapse to single-line summary with line ref
        if o["type"] == "meta" and o.get("content", []):
            c0 = o["content"][0]
            if c0.startswith(">>>tool_call"):
                # Hide noise tools (internal bookkeeping)
                tool_name = c0.split()[1].split(":")[0] if len(c0.split()) > 1 else ""
                if tool_name in _BRIEF_HIDE_TOOLS:
                    o["content_brief"] = None
                    continue
                summary = o.get("_tool_summary", f"* unknown")
                s = o.get("start_line")
                e = o.get("end_line")
                for j in (idx + 1, idx + 2):
                    if j < len(ir) and ir[j].get("_blk") == o.get("_blk"):
                        je = ir[j].get("end_line")
                        if je is not None:
                            e = je
                if s is not None and e is not None:
                    # Extract short_tid and look up tool_result range
                    parts = c0.split()[1].split(":") if len(c0.split()) > 1 else []
                    stid = parts[-1] if len(parts) >= 2 else ""
                    rr = tid_ranges.get(stid)
                    if rr:
                        summary = f"{summary} ({short}:{s+1}-{e+1},{rr[0]+1}-{rr[1]+1})"
                    else:
                        summary = f"{summary} ({short}:{s+1}-{e+1})"
                o["content_brief"] = [summary]
                continue
            if c0 == "<<<tool_call":
                o["content_brief"] = None
                continue

        if o["type"] == "tool_call":
            o["content_brief"] = None
            continue

        # meta_header → copy
        if o["type"] == "meta_header":
            o["content_brief"] = list(o["content"])
            continue

        # meta → copy
        if o["type"] == "meta":
            o["content_brief"] = list(o["content"])
            continue

        # Truncatable content
        if _is_truncatable(o):
            node_start = o.get("start_line", 0) + 1
            node_end = o.get("end_line", o.get("start_line", 0)) + 1
            ref = f"{short}:{node_start}-{node_end}"
            text = "\n".join(o["content"])
            if o["type"] == "user":
                text = _strip_noise_xml(text)
                if not text.strip():
                    o["content_brief"] = None
                    continue
            lim = truncate_user if o["type"] == "user" else truncate
            lines = (_trunc(text, lim, ref) if lim else text).split("\n")
            # Strip leading blank lines
            while lines and not lines[0]:
                lines.pop(0)
            o["content_brief"] = lines
            continue

        # Non-truncatable (images, documents, etc) → copy
        o["content_brief"] = list(o["content"])


# ── lowering: files ──

def _tool_errors(ir):
    """Map short_tid -> tool name for every tool call whose result is an error."""
    errs = {}
    for o in ir:
        if o["type"] == "meta_header" and o["content"] and o["content"][0].startswith("[tool_error]"):
            h = o["content"][0][len("[tool_error]"):].strip()
            name, _, tid = h.rpartition(":")
            errs[tid] = name or "unknown"
    return errs

def _blk_ranges(ir):
    """Map blk -> (first_line, last_line), 0-based, over all nodes of the block."""
    out = {}
    for o in ir:
        b, sl, el = o.get("_blk"), o.get("start_line"), o.get("end_line")
        if b is None or sl is None or el is None:
            continue
        r = out.get(b)
        out[b] = (sl, el) if r is None else (min(r[0], sl), max(r[1], el))
    return out

def _call_ref(o, blk_ranges, tid_ranges):
    """'call_start-call_end,result_start-result_end' (1-based) for a tool_call meta node."""
    rng = blk_ranges.get(o.get("_blk"))
    ref = f"{rng[0]+1}-{rng[1]+1}" if rng else "?"
    rr = tid_ranges.get(o.get("_tool_tid", ""))
    if rr:
        ref += f",{rr[0]+1}-{rr[1]+1}"
    return ref

def _disp_path(p, cwd):
    if cwd:
        base = cwd.rstrip("/\\")
        if p.startswith(base + os.sep) or p.startswith(base + "/"):
            return p[len(base) + 1:]
    return p

def lower_files(ir, filename="", cwd=None):
    """Files view: every file touched by a file tool, with pointers into the full view.

    Deterministic (no LLM). Lists, per file, each Read/Write/Edit call with its
    (call, result) line ranges, newest-touched file first, and marks files that a
    successful write tool modified. Returns a list of lines, or None if no file was touched.
    """
    short = _short(filename)
    tid_ranges = _tid_result_ranges(ir)
    err_tids = _tool_errors(ir)
    blk_range = _blk_ranges(ir)

    files = {}   # path -> {"ops": [...], "last": idx, "modified": bool}
    for idx, o in enumerate(ir):
        path = o.get("_tool_file")
        if not path:
            continue
        name, tid = o.get("_tool_name", "?"), o.get("_tool_tid", "")
        ref = _call_ref(o, blk_range, tid_ranges)
        err = tid in err_tids
        f = files.setdefault(path, {"ops": [], "last": idx, "last_op": name,
                                    "last_ts": None, "modified": False})
        f["ops"].append(f"  {name:<12} ({short}:{ref})" + (" error" if err else ""))
        f["last"], f["last_op"], f["last_ts"] = idx, name, o.get("_ts")
        if name in _FILE_WRITE_TOOLS and not err:
            f["modified"] = True
    if not files:
        return None

    nmod = sum(1 for f in files.values() if f["modified"])
    out = [f"[files] {len(files)} file(s) touched, {nmod} modified "
           f"(M = modified by a write tool here, R = read only)"]
    if cwd:
        out.append(f"cwd: {cwd}")
    out.append(f"Pointers are (call, result) line ranges in {short}. Newest-touched first.")
    out.append("Changes made via Bash or outside this session are NOT tracked: re-read files before relying on them.")
    for path, f in sorted(files.items(), key=lambda kv: -kv[1]["last"]):
        ts = _fmt_ts(f["last_ts"])
        out.append("")
        out.append(f"{'M' if f['modified'] else 'R'} {_disp_path(path, cwd)}  "
                   f"(last: {f['last_op']}" + (f" {ts}" if ts else "") + ")")
        out.extend(f["ops"])
    out.append("")
    return out


# ── lowering: graph ──

def build_turns(ir):
    """Group the IR into user turns: each visible user message plus the work it triggered.

    A turn is a [user] section that brief mode would show (harness noise is not a turn),
    followed by every section up to the next turn. Work done before the first user
    message becomes turn 0. Returns a list of dicts (0-based line numbers).
    """
    from collections import OrderedDict
    roles = _sec_roles(ir)
    by_sec = _searchable_by_sec(ir)
    sec_lines, sec_header = {}, {}
    for o in ir:
        s, sl, el = o.get("_sec"), o.get("start_line"), o.get("end_line")
        if s is None or sl is None:
            continue
        r = sec_lines.get(s)
        sec_lines[s] = (sl, el) if r is None else (min(r[0], sl), max(r[1], el))
        if o["type"] == "meta_header":
            sec_header[s] = o
    turn_secs = [s for s in sorted(roles) if roles[s] == "user"
                 and not _user_hidden_in_brief(ir, s, by_sec)
                 and not _section_hidden_exact(ir, s, by_sec)]

    def _new(num, sec):
        hdr = sec_header.get(sec) if sec is not None else None
        text = ""
        if sec is not None:
            raw = "\n".join("\n".join(o["content"]) for o in by_sec.get(sec, []))
            text = _strip_noise_xml(raw) or ""
            if not text and hdr is not None:   # e.g. a compact summary placeholder
                text = "\n".join(o["content"][0] for o in ir
                                 if o.get("_sec") == sec and o["type"] == "user")
        first = next((l.strip() for l in text.split("\n") if l.strip()), "")
        if len(first) > 80:
            first = first[:77] + "..."
        return {"num": num, "sec": sec, "ts": hdr.get("_ts") if hdr else None,
                "text": first, "start": None, "end": None,
                "tools": OrderedDict(), "errors": [], "files": OrderedDict()}

    # Turn boundaries (by section id): turn k spans [turn_secs[k], turn_secs[k+1]).
    turns = [_new(i + 1, s) for i, s in enumerate(turn_secs)]
    first_turn_sec = turn_secs[0] if turn_secs else None
    pre = _new(0, None)
    import bisect

    def _turn_for(sec):
        if first_turn_sec is None or sec < first_turn_sec:
            return pre
        return turns[bisect.bisect_right(turn_secs, sec) - 1]

    for s in sorted(sec_lines):
        t = _turn_for(s)
        sl, el = sec_lines[s]
        t["start"] = sl if t["start"] is None else min(t["start"], sl)
        t["end"] = el if t["end"] is None else max(t["end"], el)
        if t["ts"] is None and s in sec_header:   # turn 0 has no user header: use its first section
            t["ts"] = sec_header[s].get("_ts")

    tid_ranges = _tid_result_ranges(ir)
    errs = _tool_errors(ir)
    blk_range = _blk_ranges(ir)
    for o in ir:
        name = o.get("_tool_name")
        if not name or o.get("_sec") is None:
            continue
        t = _turn_for(o["_sec"])
        if name not in _BRIEF_HIDE_TOOLS:
            t["tools"][name] = t["tools"].get(name, 0) + 1
        tid = o.get("_tool_tid", "")
        err = tid in errs
        if err:
            rr = tid_ranges.get(tid)
            t["errors"].append((name, f"{rr[0]+1}-{rr[1]+1}" if rr else "?"))
        path = o.get("_tool_file")
        if path:
            t["files"].setdefault(path, []).append((name, err, _call_ref(o, blk_range, tid_ranges)))
    if pre["tools"] or pre["errors"]:
        turns.insert(0, pre)
    return turns

def _file_status(ops):
    return "M" if any(n in _FILE_WRITE_TOOLS and not e for n, e, _ in ops) else "R"

def _ops_str(ops):
    """'Read(error), Edit x4': consecutive identical operations are collapsed."""
    runs = []
    for n, e, _ in ops:
        key = n + ("(error)" if e else "")
        if runs and runs[-1][0] == key:
            runs[-1][1] += 1
        else:
            runs.append([key, 1])
    return ", ".join(k + (f" x{c}" if c > 1 else "") for k, c in runs)

def lower_graph(ir, filename="", cwd=None, turns=None):
    """Graph view: user turns linked to the files they touched and to earlier turns.

    Nodes are turns (T) and files (F). Edges: turn -> file (with the operations),
    file -> previous turn on that file ("prev"), and per-file lists of all turns.
    Every node carries a pointer into the full view. Returns a list of lines.
    """
    short = _short(filename)
    turns = build_turns(ir) if turns is None else turns
    if not turns:
        return None
    last_turn_on = {}          # path -> num of last turn that touched it
    file_turns = {}            # path -> [(num, ops)]
    nerr = sum(len(t["errors"]) for t in turns)
    nfiles_mod = set()
    body = []
    for t in turns:
        ts = _fmt_ts(t["ts"])
        rng = f"{short}:{t['start']+1}-{t['end']+1}" if t["start"] is not None else short
        label = f'"{t["text"]}"' if t["num"] else "(before the first user message)"
        body.append("")
        body.append(f"T{t['num']}" + (f" {ts}" if ts else "") + f" ({rng}) {label}")
        if t["tools"]:
            body.append("  tools: " + ", ".join(
                n + (f" x{c}" if c > 1 else "") for n, c in t["tools"].items()))
        if t["errors"]:
            body.append("  errors: " + ", ".join(f"{n} ({short}:{r})" for n, r in t["errors"]))
        for path, ops in t["files"].items():
            st = _file_status(ops)
            if st == "M":
                nfiles_mod.add(path)
            prev = last_turn_on.get(path)
            body.append(f"  {st} {_disp_path(path, cwd)}: {_ops_str(ops)}"
                        + (f"  [prev T{prev}]" if prev is not None else ""))
            last_turn_on[path] = t["num"]
            file_turns.setdefault(path, []).append((t["num"], ops))

    head = [f"[graph] {sum(1 for t in turns if t['num'])} turn(s), {len(file_turns)} file(s) "
            f"({len(nfiles_mod)} modified), {nerr} tool error(s). Pointers into {short}."]
    if cwd:
        head.append(f"cwd: {cwd}")
    head.append("T = user turn (its message and the work it triggered). M/R = modified/read in that turn.")
    head.append("[prev Tn] = the last earlier turn that touched the same file. Bash file changes are not tracked.")
    tail = []
    if file_turns:
        tail = ["", "[files -> turns]"]
        for path, lst in sorted(file_turns.items(), key=lambda kv: -kv[1][-1][0]):
            tail.append(f"F {_disp_path(path, cwd)}: " + "  ".join(
                f"T{n}({_ops_str(ops)})" for n, ops in lst))
    return head + body + tail + [""]

def graph_index(entries):
    """Cross-conversation file index: for each file, every turn in every conversation.

    entries: list of (full_view_path, cwd, turns). Newest activity first.
    """
    cwds = {c for _, c, _ in entries if c}
    common = cwds.pop() if len(cwds) == 1 else None
    by_file = {}
    for fp, cwd, turns in entries:
        short = _rel_path(fp)
        for t in turns:
            for path, ops in t["files"].items():
                by_file.setdefault(path, []).append((t["ts"] or "", short, t, ops))
    if not by_file:
        return []
    out = [f"[graph index] {len(by_file)} file(s) across {len(entries)} conversation(s), "
           "newest activity first" + (f"; paths relative to {common}" if common else "")]
    for path, hits in sorted(by_file.items(), key=lambda kv: max(h[0] for h in kv[1]), reverse=True):
        hits.sort(key=lambda h: h[0], reverse=True)
        out.append("")
        out.append(f"{_file_status([op for h in hits for op in h[3]])} {_disp_path(path, common)}")
        for ts, short, t, ops in hits:
            rng = f"{short}:{t['start']+1}-{t['end']+1}" if t["start"] is not None else short
            out.append(f"  {_fmt_ts(ts) or '?':<17}  {_ops_str(ops):<24} ({rng}) T{t['num']} \"{t['text']}\"")
    return out


# ── lowering: view ──

def lower_view(ir, filename="", grep_pattern=None, keep=None):
    if not grep_pattern:
        # No grep: view is same as truncated (shouldn't normally be called)
        for o in ir:
            o["content_view"] = o.get("content_brief")
        return

    short = _short(filename)

    # Per-node match check
    def _node_matches(o):
        if not o["searchable"]:
            return False
        if keep is not None and not keep(o):
            return False
        for line in o["content"]:
            if grep_pattern.search(line):
                return True
        return False

    # Pass 1: determine visibility for each searchable block
    block_visible = {}  # blk -> bool
    for o in ir:
        blk = o.get("_blk")
        if blk is None or blk in block_visible:
            continue
        if o["searchable"] and _node_matches(o):
            block_visible[blk] = True

    # Derive which sections have any visible block (for header/separator logic)
    sec_has_visible = set()
    for o in ir:
        blk = o.get("_blk")
        if blk is not None and block_visible.get(blk):
            s = o.get("_sec")
            if s is not None:
                sec_has_visible.add(s)

    # Pass 2: set content_view for each node
    for idx, o in enumerate(ir):
        s = o.get("_sec")
        blk = o.get("_blk")

        # Separator: show only between two sections that have visible blocks
        if s is None and o["type"] == "meta" and SEP in o.get("content", []):
            next_vis = False
            for j in range(idx + 1, len(ir)):
                ns = ir[j].get("_sec")
                if ns is not None:
                    next_vis = ns in sec_has_visible
                    break
            prev_vis = False
            seen = set()
            for j in range(idx - 1, -1, -1):
                ps = ir[j].get("_sec")
                if ps is not None and ps not in seen:
                    seen.add(ps)
                    if ps in sec_has_visible:
                        prev_vis = True
                        break
            o["content_view"] = list(o["content"]) if (next_vis and prev_vis) else None
            continue

        # meta_header: show if section has any visible block
        if o["type"] == "meta_header":
            o["content_view"] = list(o["content"]) if s in sec_has_visible else None
            continue

        # Thinking / tool_call metas: show if same blk matched
        if o["type"] == "meta" and o.get("content", []):
            c0 = o["content"][0]
            if c0.startswith(">>>thinking") or c0.startswith("<<<thinking") or \
               c0.startswith(">>>redacted_thinking") or c0.startswith("<<<redacted_thinking") or \
               c0.startswith(">>>tool_call ") or c0 == "<<<tool_call":
                o["content_view"] = list(o["content"]) if block_visible.get(blk) else None
                continue

        # Other meta → show if section has visible blocks
        if o["type"] == "meta":
            o["content_view"] = list(o["content"]) if s in sec_has_visible else None
            continue

        # Searchable content blocks: show only if this block matches
        if o["searchable"]:
            if _node_matches(o):
                node_start = o.get("start_line", 0) + 1
                o["content_view"] = match_lines(
                    o["content"], grep_pattern, short, node_start)
            else:
                o["content_view"] = None
            continue

        # Non-searchable (images, docs, etc) → hide
        o["content_view"] = None


# ── codegen ──

def emit(ir, key="content"):
    lines = []
    for o, c, blank in _walk(ir, key):
        if blank: lines.append("")
        lines.extend(c)
    return lines


# ── grep ──

def _rel_path(fp):
    try:
        return os.path.relpath(fp)
    except ValueError:
        return os.path.abspath(fp)

def make_block_filter(roles=None, since=None, until=None):
    """Return keep(o) -> bool for --role/--since/--until, or None when no filter is set.

    A role matches a block type exactly or as a prefix: `tool` matches tool_call,
    tool_result and tool_error; `user` also matches user_image / user_document.
    With a time bound, blocks without a timestamp are excluded.
    """
    if not roles and since is None and until is None:
        return None
    roles = tuple(roles or ())

    def keep(o):
        t = o["type"]
        if roles and not any(t == r or t.startswith(r + "_") for r in roles):
            return False
        if since is not None or until is not None:
            d = _parse_ts(o.get("_ts"))
            if d is None:
                return False
            if since is not None and d < since:
                return False
            if until is not None and d >= until:
                return False
        return True
    return keep

# ── ranking (BM25) ──

_BM25_K1, _BM25_B = 1.2, 0.75

def bm25_terms(pattern_str):
    """Query terms for ranking, taken from the literal words of a --grep regex.

    `captcha|recaptcha|bot.?detect` -> ['captcha', 'recaptcha', 'bot', 'detect'].
    Regex escapes (\\b, \\d, ...), inline flags and counted repeats are ignored.
    """
    s = re.sub(r"\(\?[a-zA-Z]+\)", " ", pattern_str)     # inline flags like (?i)
    s = re.sub(r"\\[a-zA-Z]", " ", s)                    # \b \d \w \s ...
    s = re.sub(r"\{\d*,?\d*\}", " ", s)                  # {2,5}
    seen, terms = set(), []
    for t in re.findall(r"\w{2,}", s):
        t = t.lower()
        if t not in seen:
            seen.add(t)
            terms.append(t)
    return terms

def bm25_scores(docs, candidates, terms):
    """Okapi BM25 for `candidates` (indices into docs) over the corpus `docs`.

    docs are lowercased block texts. Term frequency counts substring occurrences
    (like the regex grep, and it works for scripts without spaces such as CJK).
    Document length is measured in characters.
    """
    import math
    n = len(docs)
    if not n or not terms:
        return {i: 0.0 for i in candidates}
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df = {t: sum(1 for d in docs if t in d) for t in terms}
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}
    out = {}
    for i in candidates:
        d = docs[i]
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * len(d) / avgdl)
        sc = 0.0
        for t in terms:
            tf = d.count(t)
            if tf:
                sc += idf[t] * tf * (_BM25_K1 + 1) / (tf + norm)
        out[i] = sc
    return out

def grep_search(results, pattern, keep=None, limit=0, rank=False):
    """Print matching blocks, newest first, or by BM25 relevance with rank=True.

    limit>0 caps the number of blocks printed and reports how many were left out.
    """
    hits, docs, cand = [], [], []
    for entry in reversed(results):
        filepath, ir = entry[0], entry[1]
        short = _rel_path(filepath)
        for o in reversed(ir):
            if not o["searchable"]: continue
            if keep is not None and not keep(o): continue
            if rank:
                docs.append("\n".join(o["content"]).lower())
            lines = match_lines(o["content"], pattern, short, o.get("start_line", 0) + 1)
            if len(lines) <= 1:
                continue
            hits.append((o, lines))
            cand.append(len(docs) - 1)
    scores = None
    if rank:
        sc = bm25_scores(docs, cand, bm25_terms(pattern.pattern))
        scores = [sc[i] for i in cand]
        order = sorted(range(len(hits)), key=lambda k: -scores[k])  # stable: ties stay newest-first
        hits = [hits[k] for k in order]
        scores = [scores[k] for k in order]
    shown = hits if not limit else hits[:limit]
    for k, (o, lines) in enumerate(shown):
        if k: print()
        ts = _fmt_ts(o.get("_ts"))
        print(f"{lines[0]} [{o['type']}]" + (f" {ts}" if ts else "")
              + (f" bm25={scores[k]:.2f}" if scores is not None else ""))
        for lt in lines[1:]:
            print(lt)
    hidden = len(hits) - len(shown)
    if hidden:
        what = "lower-ranked" if rank else "older"
        print(f"\n... {hidden} {what} matching block(s) not shown (--limit {limit}); "
              f"narrow with --since/--until/--role or raise --limit")


# ── compile ──

def compile_pass(input_path, output_dir=None, truncate=128, truncate_user=256,
            grep_pattern=None, quiet=False, keep=None, graph=False):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    recs = merge_chunks(lex(input_path))
    chains = split_chains(recs)
    if not chains:
        if not quiet:
            print("No conversation chains found.")
        return []

    results, paths = [], []

    for i, chain in enumerate(chains):
        sfx = f"_{i+1}" if len(chains) > 1 else ""
        ffn = f"{base}{sfx}.txt"
        mfn = f"{base}{sfx}.min.txt"
        vfn = f"{base}{sfx}.view.txt"
        filesfn = f"{base}{sfx}.files.txt"
        graphp = os.path.join(output_dir, f"{base}{sfx}.graph.txt")
        fp = os.path.join(output_dir, ffn)
        mp = os.path.join(output_dir, mfn)
        vp = os.path.join(output_dir, vfn)
        filesp = os.path.join(output_dir, filesfn)
        data_ctr = [0]

        ir = parse(chain, output_dir, f"{base}{sfx}", data_ctr)
        assign_lines(ir)
        lower_brief(ir, truncate, ffn, truncate_user)

        full = emit(ir, "content")
        brief = emit(ir, "content_brief")

        stats_footer = _collect_stats(chain)
        if stats_footer:
            full.extend([""] + stats_footer)

        with open(fp, "w", encoding="utf-8") as f: f.write("\n".join(full))
        with open(mp, "w", encoding="utf-8") as f: f.write("\n".join(brief))

        cwd = next((r["cwd"] for r in chain if isinstance(r.get("cwd"), str)), None)
        files_view = lower_files(ir, ffn, cwd)
        if files_view:
            with open(filesp, "w", encoding="utf-8") as f: f.write("\n".join(files_view))
        elif os.path.exists(filesp):
            os.remove(filesp)  # stale view from an earlier compile

        turns = None
        if graph:
            turns = build_turns(ir)
            with open(graphp, "w", encoding="utf-8") as f:
                f.write("\n".join(lower_graph(ir, ffn, cwd, turns) or ["[graph] empty"]))

        if grep_pattern:
            lower_view(ir, ffn, grep_pattern, keep)
            view = emit(ir, "content_view")
            with open(vp, "w", encoding="utf-8") as f: f.write("\n".join(view))

        ft, bt = "\n".join(full), "\n".join(brief)
        _cnt = lambda s: sum(1 for t in _tokenize(s) if t.strip())
        results.append((fp, ir, cwd, turns))
        paths.append((fp, mp, vp if grep_pattern else None,
                       len(full), _cnt(ft), len(brief), _cnt(bt),
                       filesp if files_view else None, graphp if graph else None))

    if not quiet:
        for fp, _, _, fl, fw, *_ in paths:
            print(f"  {fp}  ({fl} lines, {fw} words)")
        for _, mp, _, _, _, bl, bw, *_ in paths:
            print(f"  {mp}  ({bl} lines, {bw} words)")
        for *_, xp, gp in paths:
            for extra in (xp, gp):
                if extra:
                    print(f"  {extra}")
        if grep_pattern:
            for _, _, vp, *_ in paths:
                if vp:
                    print(f"  {vp}")

    return results

# ── main ──

def _expand_inputs(raw):
    files = []
    for r in raw:
        expanded = globmod.glob(r, recursive=True)
        expanded.sort(key=lambda f: os.path.getmtime(f))
        files.extend(expanded if expanded else [r])
    return files

def main():
    p = argparse.ArgumentParser(description="VCC - View-oriented Conversation Compiler")
    p.add_argument("input", nargs="+")
    p.add_argument("-o", "--output-dir")
    p.add_argument("-t", "--truncate", nargs="?", type=int, const=128, default=128, metavar="N")
    p.add_argument("-tu", "--truncate-user", nargs="?", type=int, const=256, default=256, metavar="N")
    p.add_argument("--grep", metavar="PATTERN")
    p.add_argument("--role", metavar="ROLES",
                   help="with --grep: only search these block types, comma-separated "
                        "(user, assistant, thinking, tool_call, tool_result, tool_error, system; "
                        "a prefix like 'tool' matches all tool_* types)")
    p.add_argument("--since", metavar="WHEN",
                   help="with --grep: only blocks at/after this UTC time (YYYY-MM-DD or ISO-8601)")
    p.add_argument("--until", metavar="WHEN",
                   help="with --grep: only blocks before this UTC time (a bare date includes that day)")
    p.add_argument("--limit", type=int, default=0, metavar="N",
                   help="with --grep: print at most N matching blocks, newest first (0 = no limit)")
    p.add_argument("--rank", action="store_true",
                   help="with --grep: order hits by BM25 relevance to the pattern's words "
                        "instead of newest first (use an alternation of synonyms as the query)")
    p.add_argument("--graph", action="store_true",
                   help="also write .graph.txt (user turns <-> files touched); with several "
                        "conversations, print a cross-conversation file index")
    a = p.parse_args()
    try:
        a.grep = re.compile(a.grep) if a.grep else None
    except re.error as e:
        p.error(f"invalid regex for --grep: {e}")
    if (a.role or a.since or a.until or a.limit or a.rank) and not a.grep:
        p.error("--role/--since/--until/--limit/--rank require --grep (use --grep . to match everything)")
    try:
        since = _parse_when(a.since) if a.since else None
        until = _parse_when(a.until, end=True) if a.until else None
    except ValueError as e:
        p.error(str(e))
    roles = [r.strip() for r in a.role.split(",") if r.strip()] if a.role else None
    keep = make_block_filter(roles, since, until)
    all_results = []
    for f in _expand_inputs(a.input):
        # A log last written before --since cannot contain newer blocks: skip compiling it.
        if since is not None and os.path.exists(f) and os.path.getmtime(f) < since.timestamp():
            continue
        res = compile_pass(f, a.output_dir, a.truncate, a.truncate_user,
                      a.grep, quiet=bool(a.grep), keep=keep, graph=a.graph)
        all_results.extend(res)
    if a.graph and len(all_results) > 1:
        idx = graph_index([(fp, cwd, turns) for fp, _, cwd, turns in all_results])
        if idx:
            print()
            print("\n".join(idx))
    if a.grep:
        if a.graph and len(all_results) > 1:
            print()
        grep_search(all_results, a.grep, keep, a.limit, a.rank)

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
