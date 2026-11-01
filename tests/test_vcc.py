"""Tests for VCC. Run from the repo root:  python -m unittest discover tests"""

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "skills", "conversation-compiler", "scripts")
sys.path.insert(0, SCRIPTS)
import VCC  # noqa: E402

CWD = "/work/proj"


def _records():
    """A small Claude Code-style session: read (fails), write, edit, read another file."""
    def user(ts, content, **kw):
        return {"type": "user", "timestamp": ts, "cwd": CWD,
                "message": {"role": "user", "content": content}, **kw}

    def asst(ts, mid, blocks):
        return {"type": "assistant", "timestamp": ts, "cwd": CWD,
                "message": {"id": mid, "role": "assistant", "content": blocks}}

    def call(tid, name, inp):
        return {"type": "tool_use", "id": tid, "name": name, "input": inp}

    def result(tid, text, err=False):
        return {"type": "tool_result", "tool_use_id": tid, "content": text, "is_error": err}

    return [
        user("2026-04-01T09:00:00Z", "Write a pets module with a dog named Buddy."),
        asst("2026-04-01T09:00:05Z", "m1", [
            {"type": "thinking", "thinking": "Check for an existing file first."},
            {"type": "text", "text": "Let me check."},
            call("toolu_000000aaaaaa", "Read", {"file_path": CWD + "/src/pets.py"})]),
        user("2026-04-01T09:00:06Z", [result("toolu_000000aaaaaa", "Error: file not found", err=True)]),
        asst("2026-04-01T09:00:10Z", "m2", [
            call("toolu_000000bbbbbb", "Write",
                 {"file_path": CWD + "/src/pets.py", "content": "class Dog:\n    name = 'Buddy'\n"})]),
        user("2026-04-01T09:00:11Z", [result("toolu_000000bbbbbb", "File written successfully.")]),
        user("2026-04-03T15:30:00Z", "Rename the dog to Rex, and look at the README."),
        asst("2026-04-03T15:30:04Z", "m3", [
            call("toolu_000000cccccc", "Edit",
                 {"file_path": CWD + "/src/pets.py", "old_string": "Buddy", "new_string": "Rex"}),
            call("toolu_000000dddddd", "Read", {"file_path": "/elsewhere/README.md"})]),
        user("2026-04-03T15:30:05Z", [result("toolu_000000cccccc", "Edited."),
                                      result("toolu_000000dddddd", "# Readme\nThe dog is great.")]),
        asst("2026-04-03T15:30:09Z", "m4", [{"type": "text", "text": "Renamed the dog to Rex."}]),
    ]


class VCCTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.jsonl = os.path.join(self.dir, "session.jsonl")
        with open(self.jsonl, "w", encoding="utf-8") as f:
            for r in _records():
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, name):
        with open(os.path.join(self.dir, name), encoding="utf-8") as f:
            return f.read().split("\n")

    def compile(self, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = VCC.compile_pass(self.jsonl, quiet=True, **kw)
        return res

    def grep(self, pattern, keep=None, limit=0):
        res = self.compile(grep_pattern=re.compile(pattern), keep=keep)
        buf = io.StringIO()
        with redirect_stdout(buf):
            VCC.grep_search(res, re.compile(pattern), keep, limit)
        return buf.getvalue()


class TestTimestamps(VCCTestCase):
    def test_user_headers_carry_utc_time(self):
        full = self.read_after_compile("session.txt")
        self.assertIn("[user] 2026-04-01 09:00Z", full)
        self.assertIn("[user] 2026-04-03 15:30Z", full)
        brief = self.read("session.min.txt")
        self.assertIn("[user] 2026-04-01 09:00Z", brief)

    def test_header_annotation_does_not_shift_line_numbers(self):
        # The timestamp is appended to the existing header line, never a new line.
        self.compile()
        full = self.read("session.txt")
        self.assertEqual(full[0], "[user] 2026-04-01 09:00Z")
        self.assertEqual(full[2], "Write a pets module with a dog named Buddy.")

    def test_grep_hits_are_dated(self):
        out = self.grep("Rex")
        self.assertRegex(out, r"\[user\] 2026-04-03 15:30Z")
        self.assertRegex(out, r"\[assistant\] 2026-04-03 15:30Z")

    def read_after_compile(self, name):
        self.compile()
        return self.read(name)


class TestSearchFilters(VCCTestCase):
    def test_role_filter(self):
        out = self.grep(".", keep=VCC.make_block_filter(roles=["tool_error"]))
        self.assertIn("[tool_error]", out)
        self.assertIn("file not found", out)
        self.assertNotIn("[user]", out)
        self.assertNotIn("[tool_result]", out)

    def test_role_prefix_matches_all_tool_blocks(self):
        out = self.grep(".", keep=VCC.make_block_filter(roles=["tool"]))
        for t in ("[tool_call]", "[tool_result]", "[tool_error]"):
            self.assertIn(t, out)
        self.assertNotIn("[assistant]", out)

    def test_since_until(self):
        since = VCC._parse_when("2026-04-02")
        out = self.grep("dog", keep=VCC.make_block_filter(since=since))
        self.assertIn("Rename the dog", out)
        self.assertNotIn("Write a pets module", out)
        until = VCC._parse_when("2026-04-01", end=True)  # bare date includes the whole day
        out = self.grep("dog", keep=VCC.make_block_filter(until=until))
        self.assertIn("Write a pets module", out)
        self.assertNotIn("Rename the dog", out)

    def test_view_respects_filter(self):
        keep = VCC.make_block_filter(roles=["user"])
        self.compile(grep_pattern=re.compile("dog"), keep=keep)
        view = "\n".join(self.read("session.view.txt"))
        self.assertIn("Rename the dog", view)
        self.assertNotIn("Renamed the dog to Rex", view)  # assistant text filtered out

    def test_limit_shows_newest_and_reports_rest(self):
        out = self.grep("dog", limit=1)
        self.assertIn("Renamed the dog to Rex", out)       # newest block first
        self.assertNotIn("Write a pets module", out)
        self.assertRegex(out, r"\.\.\. \d+ older matching block\(s\) not shown \(--limit 1\)")

    def test_no_filter_is_unchanged(self):
        self.assertIsNone(VCC.make_block_filter())


class TestFilesView(VCCTestCase):
    def test_lists_files_with_status(self):
        self.compile()
        files = self.read("session.files.txt")
        text = "\n".join(files)
        self.assertIn("2 file(s) touched, 1 modified", text)
        self.assertIn(f"cwd: {CWD}", text)
        self.assertIn("M src/pets.py  (last: Edit 2026-04-03 15:30Z)", text)  # relative to cwd
        self.assertIn("R /elsewhere/README.md", text)                           # outside cwd
        # The failed Read is flagged.
        self.assertTrue(any(l.strip().startswith("Read") and l.endswith(" error") for l in files))

    def test_pointers_resolve_to_full_view(self):
        self.compile()
        full = self.read("session.txt")
        files = self.read("session.files.txt")
        ptr = re.compile(r"^\s+(\w+)\s+\(session\.txt:(\d+)-(\d+),(\d+)-(\d+)\)")
        checked = 0
        for line in files:
            m = ptr.match(line)
            if not m:
                continue
            name, cs, ce, rs, re_ = m.group(1), *map(int, m.groups()[1:])
            self.assertTrue(full[cs - 1].startswith(f">>>tool_call {name}:"), full[cs - 1])
            self.assertEqual(full[ce - 1], "<<<tool_call")
            # Result ranges start at the section header, as in .min.txt (README: example.txt:23-25).
            self.assertTrue(full[rs - 1].startswith(("[tool]", "[tool_error]")), full[rs - 1])
            self.assertTrue(full[rs - 1].endswith(full[cs - 1].split(":")[-1]))  # same tool id
            checked += 1
        self.assertEqual(checked, 4)

    def test_no_file_tools_no_view(self):
        with open(self.jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "user", "timestamp": "2026-04-01T09:00:00Z",
                                "message": {"role": "user", "content": "hi"}}) + "\n")
        self.compile()
        self.assertFalse(os.path.exists(os.path.join(self.dir, "session.files.txt")))


class TestCLI(VCCTestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, os.path.join(SCRIPTS, "VCC.py"), self.jsonl, *args],
                              capture_output=True, text=True, cwd=self.dir)

    def test_filters_require_grep(self):
        r = self.run_cli("--limit", "3")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("require --grep", r.stderr)

    def test_bad_time(self):
        r = self.run_cli("--grep", "x", "--since", "yesterday")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot parse time", r.stderr)

    def test_role_and_limit(self):
        r = self.run_cli("--grep", ".", "--role", "tool_error", "--limit", "5")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[tool_error] 2026-04-01 09:00Z", r.stdout)


class TestHook(VCCTestCase):
    HOOK = os.path.join(SCRIPTS, "vcc_hook.py")

    def run_hook(self, stdin, *args):
        return subprocess.run([sys.executable, self.HOOK, *args], input=stdin,
                              capture_output=True, text=True)

    def test_compile_and_archive(self):
        sub = os.path.join(self.dir, "session", "subagents")
        os.makedirs(sub)
        with open(os.path.join(sub, "agent-1.jsonl"), "w") as f:
            f.write("{}\n")
        arch = os.path.join(self.dir, "archive")
        r = self.run_hook(json.dumps({"transcript_path": self.jsonl, "hook_event_name": "PreCompact"}),
                          "--compile", "--archive", arch)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "session.min.txt")))
        proj = os.path.basename(self.dir)
        self.assertTrue(os.path.exists(os.path.join(arch, proj, "session.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(arch, proj, "session", "subagents", "agent-1.jsonl")))

    def test_never_fails(self):
        for stdin in ("not json", '{"transcript_path": "/nope.jsonl"}', "{}"):
            r = self.run_hook(stdin, "--compile")
            self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
