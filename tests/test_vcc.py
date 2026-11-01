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


class TestBM25(VCCTestCase):
    def test_terms_from_regex(self):
        self.assertEqual(VCC.bm25_terms(r"captcha|reCAPTCHA|bot.?detect"),
                         ["captcha", "recaptcha", "bot", "detect"])
        self.assertEqual(VCC.bm25_terms(r"(?i)\bdog\b|\d{2,3}x"), ["dog"])
        # optional characters are not part of the literal term
        self.assertEqual(VCC.bm25_terms(r"tests?|files*"), ["test", "file"])
        self.assertEqual(VCC.bm25_terms(r"colou?r"), ["colo"])     # matches color and colour

    def test_rarer_terms_and_shorter_blocks_score_higher(self):
        docs = ["the dog and the cat", "the dog", "a cat", "the dog " + "filler " * 50]
        sc = VCC.bm25_scores(docs, [0, 1, 2, 3], ["dog", "cat"])
        self.assertGreater(sc[0], sc[1])       # both terms beat one term
        self.assertGreater(sc[1], sc[3])       # same term, shorter block wins
        sc2 = VCC.bm25_scores(docs + ["the dog"] * 5, [0, 2], ["dog", "cat"])
        self.assertGreater(sc2[2], 0)          # "cat" is now the rarer, heavier term

    def test_rank_orders_by_relevance_not_recency(self):
        res = self.compile(grep_pattern=re.compile("Rex|Buddy|pets"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            VCC.grep_search(res, re.compile("Rex|Buddy|pets"), rank=True)
        heads = [l for l in buf.getvalue().split("\n") if l.startswith("(")]
        scores = [float(h.rsplit("bm25=", 1)[1]) for h in heads]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn("[tool_call]", heads[0])  # the Write call mentions pets.py and Buddy

    def test_rank_with_limit_footer(self):
        res = self.compile(grep_pattern=re.compile("dog"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            VCC.grep_search(res, re.compile("dog"), limit=1, rank=True)
        self.assertIn("lower-ranked matching block(s) not shown", buf.getvalue())


class TestGraph(VCCTestCase):
    def test_turns_and_file_edges(self):
        self.compile(graph=True)
        g = self.read("session.graph.txt")
        text = "\n".join(g)
        self.assertIn("[graph] 2 turn(s), 2 file(s) (1 modified), 1 tool error(s)", text)
        self.assertIn('T1 2026-04-01 09:00Z (session.txt:1-', text)
        self.assertIn('"Write a pets module with a dog named Buddy."', text)
        self.assertIn("  M src/pets.py: Read(error), Write", text)
        self.assertIn("  M src/pets.py: Edit  [prev T1]", text)          # back-link
        self.assertIn("  R /elsewhere/README.md: Read", text)
        self.assertIn("F src/pets.py: T1(Read(error), Write)  T2(Edit)", text)
        self.assertIn("  tools: Edit, Read", text)

    def test_turn_ranges_point_at_user_header(self):
        self.compile(graph=True)
        full = self.read("session.txt")
        for line in self.read("session.graph.txt"):
            m = re.match(r"^T(\d+) .*\(session\.txt:(\d+)-(\d+)\)", line)
            if m:
                self.assertTrue(full[int(m.group(2)) - 1].startswith("[user]"))

    def test_ops_collapse(self):
        self.assertEqual(VCC._ops_str([("Edit", False, ""), ("Edit", False, ""),
                                       ("Read", True, ""), ("Edit", False, "")]),
                         "Edit x2, Read(error), Edit")

    def test_work_before_first_user_message_is_turn_zero(self):
        recs = [{"type": "assistant", "timestamp": "2026-04-01T08:59:00Z",
                 "message": {"id": "m0", "content": [
                     {"type": "tool_use", "id": "toolu_00000000zzzz", "name": "Read",
                      "input": {"file_path": "/x/a.py"}}]}},
                {"type": "user", "timestamp": "2026-04-01T08:59:01Z", "message": {"content": [
                    {"type": "tool_result", "tool_use_id": "toolu_00000000zzzz", "content": "ok"}]}}] \
            + _records()
        with open(self.jsonl, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in recs) + "\n")
        self.compile(graph=True)
        text = "\n".join(self.read("session.graph.txt"))
        self.assertIn("T0 2026-04-01 08:59Z", text)  # header timestamp of the first section
        self.assertIn("(before the first user message)", text)
        self.assertIn("[graph] 2 turn(s)", text)      # T0 is not counted as a user turn

    def test_cross_conversation_index(self):
        other = os.path.join(self.dir, "other.jsonl")
        with open(other, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "user", "timestamp": "2026-05-01T10:00:00Z", "cwd": CWD,
                                "message": {"content": "Add a cat to the pets module."}}) + "\n")
            f.write(json.dumps({"type": "assistant", "timestamp": "2026-05-01T10:00:02Z", "cwd": CWD,
                                "message": {"id": "o1", "content": [
                                    {"type": "tool_use", "id": "toolu_0000000other", "name": "Edit",
                                     "input": {"file_path": CWD + "/src/pets.py"}}]}}) + "\n")
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "VCC.py"), self.jsonl, other,
                            "--graph"], capture_output=True, text=True, cwd=self.dir)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        self.assertIn("[graph index] 2 file(s) across 2 conversation(s)", out)
        self.assertIn(f"paths relative to {CWD}", out)
        block = out.split("M src/pets.py", 1)[1].split("\n\n", 1)[0].strip().split("\n")
        self.assertIn("other.txt", block[0])        # newest conversation first
        self.assertIn('"Add a cat to the pets module."', block[0])
        self.assertIn("session.txt", block[1])
        self.assertTrue(os.path.exists(os.path.join(self.dir, "other.graph.txt")))

    def test_subagent_links(self):
        # Claude Code layout: the parent's tool result carries toolUseResult.agentId and
        # the subagent's log lives in <session>/subagents/agent-<id>.jsonl
        recs = _records()[:5] + [
            {"type": "assistant", "timestamp": "2026-04-01T09:01:00Z",
             "message": {"id": "m9", "content": [{"type": "tool_use", "id": "toolu_0000000agent",
                                                  "name": "Agent", "input": {"description": "Audit pets",
                                                                             "prompt": "..."}}]}},
            {"type": "user", "timestamp": "2026-04-01T09:02:00Z",
             "toolUseResult": {"agentId": "abc123def", "status": "completed"},
             "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_0000000agent",
                                      "content": "Audit done."}]}}]
        _write_jsonl(self.jsonl, recs)
        sub_dir = os.path.join(self.dir, "session", "subagents")
        os.makedirs(sub_dir)
        sub = os.path.join(sub_dir, "agent-abc123def.jsonl")
        _write_jsonl(sub, [{"type": "user", "agentId": "abc123def", "isSidechain": True,
                            "timestamp": "2026-04-01T09:01:01Z",
                            "message": {"content": "Audit pets"}}])
        self.compile(graph=True)
        graph = "\n".join(self.read("session.graph.txt"))
        self.assertIn("1 subagent(s)", graph)
        self.assertRegex(graph, r"  subagent abc123def \(result session\.txt:\d+\) -> "
                                r".*session/subagents/agent-abc123def\.jsonl")
        VCC.compile_pass(sub, quiet=True, graph=True)
        with open(os.path.join(sub_dir, "agent-abc123def.graph.txt")) as f:
            self.assertIn("This log is subagent abc123def.", f.read())

    def test_no_graph_by_default(self):
        self.compile()
        self.assertFalse(os.path.exists(os.path.join(self.dir, "session.graph.txt")))


class TestToolSemantics(unittest.TestCase):
    """File effects come from argument shapes, not tool names, so any harness works."""

    def files(self, name, inp):
        return dict(VCC.tool_files(name, inp))

    def test_claude_code_tools(self):
        self.assertEqual(self.files("Read", {"file_path": "/a.py"}), {"/a.py": False})
        self.assertEqual(self.files("Write", {"file_path": "/a.py", "content": "x"}), {"/a.py": True})
        self.assertEqual(self.files("Edit", {"file_path": "/a.py", "old_string": "a",
                                             "new_string": "b"}), {"/a.py": True})
        self.assertEqual(self.files("NotebookEdit", {"notebook_path": "/n.ipynb",
                                                     "new_source": "x"}), {"/n.ipynb": True})
        self.assertEqual(self.files("Grep", {"pattern": "x", "path": "/src"}), {})   # search scope
        self.assertEqual(self.files("Bash", {"command": "ls"}), {})

    def test_other_argument_shapes(self):
        # other argument names for the same effects
        self.assertEqual(self.files("read_file", {"absolute_path": "/a.py"}), {"/a.py": False})
        self.assertEqual(self.files("replace", {"file_path": "/a.py", "old_string": "a",
                                                "new_string": "b"}), {"/a.py": True})
        self.assertEqual(self.files("read_many_files", {"paths": ["/a.py", "/b.py"]}),
                         {"/a.py": False, "/b.py": False})
        # camelCase arguments, edit verb in the tool name
        self.assertEqual(self.files("edit", {"filePath": "/a.ts", "oldString": "a",
                                             "newString": "b"}), {"/a.ts": True})
        self.assertEqual(self.files("delete_file", {"path": "/old.txt"}), {"/old.txt": True})

    def test_patch_text_anywhere_in_arguments(self):
        patch = ("*** Begin Patch\n*** Update File: src/app.py\n@@\n-a\n+b\n"
                 "*** Add File: src/new.py\n+x\n*** End Patch")
        self.assertEqual(self.files("apply_patch", {"input": patch}),
                         {"src/app.py": True, "src/new.py": True})
        self.assertEqual(self.files("shell", {"command": ["apply_patch", patch]}),
                         {"src/app.py": True, "src/new.py": True})
        diff = "--- a/x.c\n+++ b/x.c\n@@ -1 +1 @@\n-a\n+b\n--- /dev/null\n+++ b/y.c\n"
        self.assertEqual(self.files("run", {"diff": diff}), {"x.c": True, "y.c": True})

    def test_summary_uses_argument_keys(self):
        self.assertEqual(VCC._tool_summary("WebSearch", {"query": "bm25"}), '* WebSearch "bm25"')
        self.assertEqual(VCC._tool_summary("shell", {"command": ["bash", "-lc", "pytest -q"]}),
                         '* shell "pytest -q"')
        files = VCC.tool_files("apply_patch", {"input": "*** Update File: a.py\n"})
        self.assertEqual(VCC._tool_summary("apply_patch", {"input": "*** Update File: a.py\n"}, files),
                         '* apply_patch "a.py"')


class TestShellWrites(unittest.TestCase):
    """Writes inferred from shell commands: conservative, quote/heredoc-aware."""

    def w(self, cmd, cwd="/w"):
        return VCC.shell_writes(cmd, cwd)

    def test_redirections_and_tee(self):
        self.assertEqual(self.w("echo hi > out.txt"), ["/w/out.txt"])
        self.assertEqual(self.w("echo 'a > b' | tee -a log.txt notes.md"), ["/w/log.txt", "/w/notes.md"])
        self.assertEqual(self.w("cmd 2>&1 >/dev/null; ls > /dev/null 2> err.log"), ["/w/err.log"])
        self.assertEqual(self.w('echo "a\nb" > multi.txt'), ["/w/multi.txt"])

    def test_commands(self):
        self.assertEqual(self.w("cd sub && sed -i 's/a/b/' x.py y.py && cd .. && touch z"),
                         ["/w/sub/x.py", "/w/sub/y.py", "/w/z"])
        self.assertEqual(self.w("sed -i.bak -e 's/a/b/' -e 's/c/d/' conf.ini"), ["/w/conf.ini"])
        self.assertEqual(self.w("git checkout -- a.py && git restore --staged c.py"), ["/w/a.py", "/w/c.py"])
        self.assertEqual(self.w("cp src/a.py backup/ && dd if=/dev/zero of=disk.img"),
                         ["/w/backup/a.py", "/w/disk.img"])
        self.assertEqual(self.w("mv old.py new/"), ["/w/old.py", "/w/new/old.py"])

    def test_reads_and_unknowns_are_ignored(self):
        self.assertEqual(self.w("sed -n '1,5p' f.txt; grep -r x . | head; cat a.txt"), [])
        self.assertEqual(self.w("rm -rf $TMP/build *.pyc ~/cache; rm stale.lock"), ["/w/stale.lock"])
        self.assertEqual(self.w('echo "unbalanced > x'), [])
        self.assertEqual(self.w("python3 - <<'PY'\nopen('hidden.txt','w')\nPY"), [])

    def test_heredoc_body_is_data(self):
        self.assertEqual(self.w("cat > new.py <<'EOF'\nprint('x > y')\nrm -rf /\nEOF\nmv new.py src/"),
                         ["/w/new.py", "/w/src/new.py"])

    def test_unknown_directory_drops_relative_paths(self):
        # regression: a quoted string spanning lines must not hide the `cd`
        self.assertEqual(self.w('cd /a && python3 -c "\nimport x\n"; rm b.txt'), ["/a/b.txt"])
        self.assertEqual(self.w('cd $S/fx; python3 -c "\nprint(1)\n"; rm big.jsonl; rm /abs/x'),
                         ["/abs/x"])
        self.assertEqual(self.w("echo x > rel.txt", cwd=None), [])

    def test_argv_lists(self):
        self.assertEqual(self.w(["bash", "-lc", "cd /repo && echo 1 >> notes.txt"], None),
                         ["/repo/notes.txt"])
        self.assertEqual(self.w(["sed", "-i", "s/x/y/", "file with space.txt"]),
                         ["/w/file with space.txt"])

    def test_marked_as_inferred_in_views(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.jsonl")
            _write_jsonl(p, [
                {"type": "user", "timestamp": "2026-04-01T09:00:00Z", "cwd": "/proj",
                 "message": {"content": "bump the version"}},
                {"type": "assistant", "timestamp": "2026-04-01T09:00:01Z", "cwd": "/proj",
                 "message": {"id": "a", "content": [{"type": "tool_use", "id": "toolu_00000000sh01",
                             "name": "Bash", "input": {"command": "sed -i 's/1.0/1.1/' setup.cfg"}}]}}])
            VCC.compile_pass(p, quiet=True, graph=True)
            with open(os.path.join(d, "s.files.txt")) as f:
                files = f.read()
            self.assertIn("M setup.cfg", files)
            self.assertIn("  Bash~ ", files)
            with open(os.path.join(d, "s.graph.txt")) as f:
                self.assertIn("M setup.cfg: Bash~", f.read())


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestChatAdapter(VCCTestCase):
    """Plain role/content JSONL (OpenAI-style tool_calls): the any-harness fallback."""

    def setUp(self):
        super().setUp()
        _write_jsonl(self.jsonl, [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Fix the bug in util.py", "timestamp": "2026-06-01T10:00:00Z",
             "cwd": "/w"},
            {"role": "assistant", "content": "Looking.", "timestamp": "2026-06-01T10:00:02Z",
             "tool_calls": [{"id": "call_1", "type": "function", "function": {
                 "name": "write_file", "arguments": json.dumps({"path": "/w/util.py", "content": "ok"})}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "written",
             "timestamp": "2026-06-01T10:00:03Z"},
            {"role": "assistant", "content": [{"type": "text", "text": "Fixed util.py."}],
             "timestamp": "2026-06-01T10:00:04Z"},
        ])

    def test_detect_and_compile(self):
        recs, profile, harness = VCC.load(self.jsonl)
        self.assertEqual(harness, "chat")
        self.compile(graph=True)
        full = "\n".join(self.read("session.txt"))
        self.assertIn("[user] 2026-06-01 10:00Z", full)
        self.assertIn(">>>tool_call write_file:call_1", full)
        self.assertIn("[tool] write_file:call_1", full)
        self.assertIn("M util.py", "\n".join(self.read("session.files.txt")))
        self.assertIn('"Fix the bug in util.py"', "\n".join(self.read("session.graph.txt")))


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

    def test_sweep_mirrors_logs_only(self):
        src = os.path.join(self.dir, "logs", "projects", "p1")
        os.makedirs(src)
        for fn in ("session-a.jsonl", "session-b.json", "notes.txt"):
            with open(os.path.join(src, fn), "w") as f:
                f.write("{}\n")
        arch = os.path.join(self.dir, "arch")
        root = os.path.join(self.dir, "logs")
        r = self.run_hook("", "--archive", arch, "--sweep", root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("2 file(s) archived", r.stderr)
        copied = sorted(os.path.relpath(os.path.join(dp, f), arch)
                        for dp, _, fs in os.walk(arch) for f in fs)
        self.assertEqual(len(copied), 2)
        self.assertTrue(all(c.endswith(("session-a.jsonl", "session-b.json")) for c in copied))
        r = self.run_hook("", "--archive", arch, "--sweep", root)
        self.assertIn("0 file(s) archived", r.stderr)                 # idempotent
        r = self.run_hook("", "--archive", os.path.join(root, "a"), "--sweep", root)
        self.assertIn("must not contain each other", r.stderr)
        self.assertEqual(r.returncode, 0)

    def test_never_fails(self):
        for stdin in ("not json", '{"transcript_path": "/nope.jsonl"}', "{}"):
            r = self.run_hook(stdin, "--compile")
            self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
