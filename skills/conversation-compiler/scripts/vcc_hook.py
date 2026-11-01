#!/usr/bin/env python3
"""Optional Claude Code hook for VCC. Reads the hook's JSON input from stdin.

  --compile        compile the transcript's views now (e.g. on PreCompact), so the
                   next /recall finds .txt/.min.txt/.files.txt already built.
  --archive DIR    copy the transcript (and its subagent logs) into DIR/<project>/,
                   so it survives Claude Code's cleanupPeriodDays deletion.

The hook never blocks Claude Code: any failure is reported on stderr and it exits 0.
For a manual run, pass --transcript PATH instead of piping hook JSON on stdin.
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


def compile_views(transcript):
    import VCC
    VCC.compile_pass(transcript, quiet=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--archive", metavar="DIR")
    p.add_argument("--transcript", metavar="PATH", help="use this JSONL instead of hook stdin")
    a = p.parse_args()

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
