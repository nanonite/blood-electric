#!/usr/bin/env python3
"""
First-pass lexical/symbolic analysis of Blood Electric (Kenji Siratori).
Implements the initial rungs of docs/meaning-extraction-methods.md:
  L1 operator inventory + transition matrix + silence-removal variants
  L2 token/neologism inventory
  L3 frequency, collocation (PMI), entity candidates
  L6 entropy, compressibility, periodicity

Usage: python analyze.py <markdown-file> [outdir]
"""
import sys, re, os, gzip, math, json, collections, datetime

# --- operators of the symbolic machine (longest first for the scanner) ---
OPS = ["<<", ">>", "//", "::", "++", "--", "==", "->", "<-", "=", "/", "<", ">", "+", "`", "^", "|", "~"]

FUNCTION = set("""a an the of to in on at by for with from into over under and or but nor so yet
is are was were be been being am do does did done have has had having will would shall should can
could may might must this that these those it its he she they them him her his their our your my
i you we me us as if then than there here when where which who whom whose what why how not no
all any both each few more most other some such only own same too very s t d ll re ve m o""".split())

def load_words():
    for p in ("/usr/share/dict/words", "/usr/share/dict/american-english", "/usr/share/dict/british-english"):
        if os.path.exists(p):
            return set(w.strip().lower() for w in open(p, errors="ignore"))
    return set()

def tokenize(text):
    """Word tokens incl. hyphen/equals/slash compounds as single units, plus their parts."""
    raw = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-=/][A-Za-z0-9]+)*", text)
    return raw

def operators(text):
    out = []
    i = 0
    while i < len(text):
        for op in OPS:
            if text.startswith(op, i):
                out.append(op); i += len(op); break
        else:
            i += 1
    return out

def entropy(items):
    if not items: return 0.0
    c = collections.Counter(items); n = len(items)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def compress_ratio(text):
    b = text.encode("utf-8")
    if not b: return 0.0
    return len(gzip.compress(b, 9)) / len(b)

def main():
    md = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(md)
    os.makedirs(outdir, exist_ok=True)
    text = open(md, errors="ignore").read()
    # strip markdown headers but remember them as section markers
    headers = re.findall(r"^##\s+(.*)$", text, flags=re.M)
    body = re.sub(r"^#+\s+.*$", "", text, flags=re.M)

    words = tokenize(body)
    low = [w.lower() for w in words]
    wc = collections.Counter(low)
    ops = operators(body)
    opc = collections.Counter(ops)
    dictset = load_words()

    L = []
    def p(s=""): L.append(s); print(s)

    p("# Blood Electric — first-pass corpus analysis")
    p(f"\n_Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from `{md}`_\n")
    p("## 1. Size & lexical statistics")
    p(f"- characters: {len(body):,}")
    p(f"- word tokens: {len(words):,}")
    p(f"- unique word types: {len(wc):,}")
    p(f"- type/token ratio: {len(wc)/max(1,len(words)):.3f}")
    hapax = [w for w,c in wc.items() if c == 1]
    p(f"- hapax legomena: {len(hapax):,} ({len(hapax)/max(1,len(wc)):.1%} of types)")
    p(f"- token entropy: {entropy(low):.3f} bits/token")
    p(f"- char entropy:  {entropy(list(body)):.3f} bits/char")
    p(f"- gzip ratio:    {compress_ratio(body):.3f} (lower = more repetitive/compressible)")
    p(f"- section headers detected: {len(headers)}")

    p("\n## 2. The symbolic machine (operators)")
    total_ops = sum(opc.values())
    p(f"- operator occurrences: {total_ops:,} ({total_ops/max(1,len(words)):.2f} per word)")
    for op, c in opc.most_common():
        p(f"    {op!r:6} {c:7,}")
    # transition matrix
    trans = collections.Counter((ops[i], ops[i+1]) for i in range(len(ops)-1))
    p("\n### Operator bigrams (top 15)")
    for (a,b), c in trans.most_common(15):
        p(f"    {a!r} -> {b!r}: {c:,}")

    p("\n## 3. Silence removal (Begus transform)")
    variants = {
        "delete_all_operators": re.sub(r"(?:<<|>>|//|::|\+\+|=|/|<|>|\+|\^|\||~|`)", " ", body),
        "operators_to_space":   re.sub(r"[<>=/:\+`^|~]+", " ", body),
        "operators_to_period":  re.sub(r"[<>=/:\+`^|~]+", ". ", body),
    }
    p("| variant | words | ttok entropy | gzip |")
    p("|---|---|---|---|")
    p(f"| original | {len(words):,} | {entropy(low):.3f} | {compress_ratio(body):.3f} |")
    for name, t in variants.items():
        w = [x.lower() for x in tokenize(t)]
        p(f"| {name} | {len(w):,} | {entropy(w):.3f} | {compress_ratio(t):.3f} |")
        open(os.path.join(outdir, f"silence_{name}.txt"), "w").write(
            re.sub(r"[ \t]+", " ", t))

    p("\n## 4. Entity / neologism candidates")
    p("(non-dictionary, non-function, length>=3, frequency>=2 — or compound with operator)")
    cand = collections.Counter()
    for w, c in wc.items():
        if w in FUNCTION or len(w) < 3: continue
        compound = bool(re.search(r"[-=/,]", w))
        base = re.split(r"[-=/,]", w)[0]
        known = base in dictset
        if compound or (w not in dictset and c >= 2):
            cand[w] = c
    for w, c in cand.most_common(60):
        p(f"    {c:5,}  {w}")

    p("\n## 5. Collocations (PMI, freq>=5, top 40)")
    bi = collections.Counter(zip(low, low[1:]))
    n = len(low); pmi = []
    for (a,b), c in bi.items():
        if c < 5 or a in FUNCTION or b in FUNCTION: continue
        denom = wc[a]*wc[b]
        if denom == 0: continue
        pmi.append((math.log2((c*n)/denom), a, b, c))
    for s,a,b,c in sorted(pmi, reverse=True)[:40]:
        p(f"    {s:5.2f}  {a} {b}  ({c:,})")

    p("\n## 6. Section headers")
    for h in headers:
        p(f"    - {h}")

    p("\n## 7. Methods not yet run in this pass")
    p("- L0 multi-engine OCR ensemble & confidence mapping")
    p("- L3 embeddings (word2vec/fastText) on corpus-only vs merged English")
    p("- L4 clause/parse/derangement curve per page")
    p("- L5 sentence-transformer clustering, topics, UMAP map, affect curve")
    p("- L6 LM perplexity per page; FFT periodicity of delimiter intervals")
    p("- L7 keyness vs cyberpunk / Burroughs / Finnegans Wake / technical corpora")
    p("- L8 interlinear gloss + termbase; L9 verification")

    report = "\n".join(L)
    open(os.path.join(outdir, "first_pass_report.md"), "w").write(report)
    p(f"\n[written] {os.path.join(outdir,'first_pass_report.md')}")

if __name__ == "__main__":
    main()
