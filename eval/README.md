# Evaluation

## Retrieval on LongMemEval_S (`longmemeval_retrieval.py`)

LongMemEval ([Wu et al., ICLR 2025](https://arxiv.org/abs/2410.10813)) gives each question a haystack of ~50 timestamped chat sessions and marks the turns that hold the evidence. This evaluation measures whether VCC's search output puts that evidence near the top. No LLM is involved, so results are deterministic and cheap to reproduce.

**Method**

- Each haystack session is written as its own chat JSONL and compiled with VCC's generic `chat` adapter, the same as searching many conversations with `/searchchat`. File order follows session dates.
- The query is the question's content words as a case-insensitive alternation, e.g. `(?i)degree|graduate`. This is deliberately weak: no paraphrases or synonyms, which an agent would normally add.
- The real `compile_pass` + `grep_search` run. Their printed hits are parsed and mapped back to (session, turn).
- The same matched blocks are ordered three ways:
  - **recency** (VCC default, newest first)
  - **coverage** (naive: count of distinct query words, to separate BM25's weighting from simple term counting)
  - **bm25** (`--rank`)
- Abstention questions (no evidence) are excluded, leaving 470 questions.

**Results** (VCC commit after `9d55db3`; LongMemEval_S cleaned; 82 s on one CPU)

| metric | recency | coverage | **bm25** |
|---|---|---|---|
| any evidence turn in top 5 | 0.081 | 0.670 | **0.815** |
| any evidence turn in top 10 | 0.196 | 0.774 | **0.879** |
| any evidence turn in top 30 | 0.557 | 0.887 | **0.932** |
| all evidence turns in top 10 | 0.125 | 0.654 | **0.787** |
| all evidence turns in top 30 | 0.397 | 0.807 | **0.871** |
| evidence sessions in top 10 | 0.233 | 0.870 | **0.927** |
| evidence turns matched by the regex at all | 0.927 | 0.927 | 0.927 |

Matched blocks per question: 109 on average. So `--rank --limit 30` keeps about 94% of the evidence the regex can find (0.871 of 0.927) in about a quarter of the output.

Any evidence turn in top 10, by question type:

| type | n | recency | coverage | bm25 |
|---|---|---|---|---|
| knowledge-update | 72 | 0.347 | 0.944 | 0.958 |
| multi-session | 121 | 0.223 | 0.752 | 0.860 |
| single-session-assistant | 56 | 0.036 | 0.893 | 0.929 |
| single-session-preference | 30 | 0.033 | 0.200 | 0.533 |
| single-session-user | 64 | 0.219 | 0.797 | 0.938 |
| temporal-reasoning | 127 | 0.181 | 0.772 | 0.882 |

**What this does and does not show**

- It shows that, for searching across many conversations, `--rank` surfaces evidence far earlier than newest-first order, and that BM25's weighting beats plain term counting. This is why the skill docs recommend `--rank --limit 30` for `/searchchat`.
- Newest-first is still the default, because right after a compaction (`/recall`) the relevant material is usually recent. LongMemEval places evidence at random points in time, which is the worst case for recency.
- LongMemEval is chat, not coding-agent traces with tool calls. Results on tool-heavy logs may differ.
- These are not comparable to retrieval numbers in the LongMemEval paper, which uses different retrieval units, indexes and retrievers.
- The BM25 parameters (k1=1.2, b=0.75) are textbook defaults and were not tuned on this data.
- Preference questions stay hard: the evidence rarely repeats the question's words, which is where an agent's synonyms in the query matter.

**Reproduce**

```bash
curl -L -o longmemeval_s_cleaned.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
python eval/longmemeval_retrieval.py longmemeval_s_cleaned.json        # --n 20 for a quick run
```

The dataset is MIT-licensed and is not included in this repo.
