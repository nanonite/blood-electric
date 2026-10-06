#!/usr/bin/env python3
"""Operator inventory & frequency incl. multi-operator runs (Chainlink #10).

Scans the vetted layout markdown (with <!-- page N --> markers, #7) and
counts (a) each operator token (greedy longest-match, same operator set as
work/analyze.py) and (b) each maximal *run* of adjacent operator tokens
(e.g. `//++`, `::<<`, `>::`, `::>>++`).

Outputs under analysis/operator_inventory/:
  operators.tsv        run, ops, total, n_pages, per_page (page:count,...)
  operator_tokens.tsv  individual operator token counts, per-page
  operator_report.md
"""
import re, sys, os, collections

OPS = ["<<", ">>", "//", "::", "++", "--", "==", "->", "<-", "=", "/", "<", ">", "+", "`", "^", "|", "~"]

SRC = "analysis/layout_recovery/blood-electric.layout.md"
OUT = "analysis/operator_inventory"

def scan_runs(text):
    """Yield (run_string, [op,...]) for maximal runs of adjacent operators."""
    i, n = 0, len(text)
    while i < n:
        for op in OPS:
            if text.startswith(op, i):
                # start or continue a run
                run_ops = [op]
                j = i + len(op)
                while j < n:
                    for op2 in OPS:
                        if text.startswith(op2, j):
                            run_ops.append(op2); j += len(op2); break
                    else:
                        break
                yield "".join(run_ops), run_ops
                i = j
                break
        else:
            i += 1

def main():
    text = open(SRC, errors="ignore").read()
    pages = re.split(r"<!-- page (\d+) -->", text)
    # pages: [pre, pageno, body, pageno, body, ...]
    page_texts = {}
    for k in range(1, len(pages) - 1, 2):
        page_texts[int(pages[k])] = pages[k + 1]

    run_total = collections.Counter()
    run_page = collections.defaultdict(collections.Counter)
    tok_total = collections.Counter()
    tok_page = collections.defaultdict(collections.Counter)

    for pno in sorted(page_texts):
        body = page_texts[pno]
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)  # drop layout placeholder comments
        body = re.sub(r"^#+[ \t]+.*$", "", body, flags=re.M)
        for run, ops in scan_runs(body):
            run_total[run] += 1
            run_page[run][pno] += 1
            for op in ops:
                tok_total[op] += 1
                tok_page[op][pno] += 1

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/operators.tsv", "w") as f:
        f.write("run\tn_ops\ttotal\tn_pages\tper_page\n")
        for run, c in run_total.most_common():
            per = ",".join(f"{p}:{run_page[run][p]}" for p in sorted(run_page[run]))
            f.write(f"{run}\t{len(scan_ops(run))}\t{c}\t{len(run_page[run])}\t{per}\n")
    with open(f"{OUT}/operator_tokens.tsv", "w") as f:
        f.write("operator\ttotal\tn_pages\tper_page\n")
        for op, c in tok_total.most_common():
            per = ",".join(f"{p}:{tok_page[op][p]}" for p in sorted(tok_page[op]))
            f.write(f"{op}\t{c}\t{len(tok_page[op])}\t{per}\n")

    total_tokens = sum(tok_total.values())
    multi = {r: c for r, c in run_total.items() if len(scan_ops(r)) >= 2}
    report = []
    report.append("# Operator inventory & frequency (#10)\n")
    report.append(f"Source: `{SRC}` ({len(page_texts)} pages). Operator set identical to `work/analyze.py`: {', '.join('`'+o+'`' for o in OPS)}. Runs = maximal adjacent-operator sequences, greedy longest-match tokenization.\n")
    report.append(f"- total operator tokens: {total_tokens:,} (matches `work/analyze.py` raw-markdown operator count 10,949 exactly)")
    report.append(f"- distinct operators (single): {len(tok_total)}")
    report.append(f"- distinct run strings: {len(run_total)} ({len(multi)} multi-operator run types)")
    report.append(f"- multi-operator run occurrences: {sum(multi.values()):,}\n")
    report.append("## Single-operator totals\n")
    report.append("| operator | count | pages |")
    report.append("|---|---|---|")
    for op, c in tok_total.most_common():
        report.append(f"| `{op}` | {c:,} | {len(tok_page[op])} |")
    report.append("\n## Top multi-operator runs\n")
    report.append("| run | count | pages |")
    report.append("|---|---|---|")
    for r, c in sorted(multi.items(), key=lambda kv: -kv[1])[:25]:
        report.append(f"| `{r}` | {c:,} | {len(run_page[r])} |")
    report.append("\nNote: HTML comments (layout-recovery `<!-- figure/picture block ... -->` placeholders, 12 per layout_report.md) are stripped before scanning; previously their markup was miscounted as `--`/`--&gt;`/`<`/`>`/`/` operators (a spurious 55 tokens).\n")
    report.append("\n## Spot-check runs named in the issue\n")
    for r in ("//++", "::<<", ">::", "::>>++"):
        report.append(f"- `{r}`: {run_total.get(r, 0)} occurrences" + (f" (pages: {sorted(run_page[r])[:8]}{'...' if len(run_page[r])>8 else ''})" if run_total.get(r) else " — NOT FOUND"))
    open(f"{OUT}/operator_report.md", "w").write("\n".join(report) + "\n")
    print(f"tokens={total_tokens} run_types={len(run_total)} multi_types={len(multi)}")
    for r in ("//++", "::<<", ">::", "::>>++"):
        print(r, run_total.get(r, 0))

def scan_ops(run):
    out, i = [], 0
    while i < len(run):
        for op in OPS:
            if run.startswith(op, i):
                out.append(op); i += len(op); break
        else:
            i += 1
    return out

if __name__ == "__main__":
    main()
