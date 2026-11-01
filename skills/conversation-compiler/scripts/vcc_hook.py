#!/usr/bin/env python3
"""Keep agent logs around and pre-build VCC views.

As a Claude Code hook (reads the hook's JSON input from stdin):
  --compile        compile the transcript's views now (e.g. on PreCompact).
  --archive DIR    copy the transcript (and its subagent logs) into DIR/<project>/,
                   so it survives Claude Code's cleanupPeriodDays deletion.
For a manual run, pass --transcript PATH instead of piping hook JSON on stdin.

For any harness (no hooks needed; e.g. from cron):
  --archive DIR --sweep SRC [--sweep SRC2 ...]
                   copy every log file (.jsonl, .json, .jsonl.zst) under SRC into
                   DIR/<SRC path relative to home>/..., only when the source is newer.
                   e.g. --sweep ~/.codex/sessions --sweep ~/.gemini/tmp

Never fails: errors go to stderr and it always exits 0.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _copy_if_newer(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)  # atomic: a reader never sees a half-written archive copy


def archive(transcript, archive_dir):
    archive_dir = os.path.abspath(os.path.expanduser(archive_dir))
    project = os.path.basename(os.path.dirname(os.path.abspath(transcript)))
    dest = os.path.join(archive_dir, project)
    _copy_if_newer(transcript, os.path.join(dest, os.path.basename(transcript)))
    # Subagent logs live in <session-id>/subagents/*.jsonl next to the transcript.
    stem = os.path.splitext(transcript)[0]
    sub = os.path.join(stem, "subagents")
    if os.path.isdir(sub):
        for fn in os.listdir(sub):
            if fn.endswith(".jsonl"):
                _copy_if_newer(os.path.join(sub, fn),
                               os.path.join(dest, os.path.basename(stem), "subagents", fn))


_LOG_EXTS = (".jsonl", ".json", ".jsonl.zst")

def sweep(src, archive_dir):
    """Mirror every log file under src into archive_dir; returns the number copied."""
    src = os.path.abspath(os.path.expanduser(src))
    archive_dir = os.path.abspath(os.path.expanduser(archive_dir))
    home = os.path.expanduser("~")
    rel_root = os.path.relpath(src, home) if src.startswith(home + os.sep) else src.lstrip(os.sep)
    dest_root = os.path.join(archive_dir, rel_root)
    if dest_root == src or dest_root.startswith(src + os.sep) or src.startswith(archive_dir + os.sep):
        raise ValueError("archive and sweep directories must not contain each other")
    n = 0
    for dirpath, _, files in os.walk(src):
        for fn in files:
            if fn.endswith(_LOG_EXTS):
                s = os.path.join(dirpath, fn)
                d = os.path.join(dest_root, os.path.relpath(s, src))
                if not (os.path.exists(d) and os.path.getmtime(d) >= os.path.getmtime(s)):
                    _copy_if_newer(s, d)
                    n += 1
    return n


def compile_views(transcript):
    import VCC
    VCC.compile_pass(transcript, quiet=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--archive", metavar="DIR")
    p.add_argument("--transcript", metavar="PATH", help="use this JSONL instead of hook stdin")
    p.add_argument("--sweep", metavar="SRC", action="append",
                   help="with --archive: mirror all logs under SRC (repeatable)")
    a = p.parse_args()

    if a.sweep:
        if not a.archive:
            print("vcc_hook: --sweep needs --archive DIR", file=sys.stderr)
            return
        for src in a.sweep:
            try:
                n = sweep(src, a.archive)
                print(f"vcc_hook: {src}: {n} file(s) archived", file=sys.stderr)
            except Exception as e:
                print(f"vcc_hook: sweep {src} failed: {e}", file=sys.stderr)
        return

    transcript = a.transcript
    if not transcript:
        try:
            transcript = json.load(sys.stdin).get("transcript_path")
        except Exception as e:
            print(f"vcc_hook: cannot read hook input: {e}", file=sys.stderr)
            return
    if not transcript or not os.path.isfile(transcript):
        print(f"vcc_hook: no transcript at {transcript!r}", file=sys.stderr)
        return

    # Archive first: it is the step that protects data.
    if a.archive:
        try:
            archive(transcript, a.archive)
        except Exception as e:
            print(f"vcc_hook: archive failed: {e}", file=sys.stderr)
    if a.compile:
        try:
            compile_views(transcript)
        except Exception as e:
            print(f"vcc_hook: compile failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.exit(0)
