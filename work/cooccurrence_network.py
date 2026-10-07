#!/usr/bin/env python3
"""#29 Co-occurrence network and community detection -> analysis/network/

Builds a semantic network over the book's **unit** layer (#17 ``units.tsv``,
markup-screened with #19's ledger and the #27 tag-run detector):

  * **nodes** = content term types (function words removed) whose frequency
    reaches ``MIN_FREQ``;
  * **edges** = PMI over a sliding ``±WINDOW``-unit co-occurrence window,
    kept when the pair co-occurs at least ``MIN_COOCCUR`` times and its PMI
    reaches ``MIN_PMI``;
  * **communities** = Louvain (``networkx``) over the weighted graph;
  * **centrality** = degree, strength (weighted degree), betweenness,
    eigenvector and PageRank, which together identify the load-bearing
    concepts.

The issue's hypothesis is that Louvain recovers the semantic domains
(body / machine / violence / sex / genetics / corporate).  The report tests
that hypothesis honestly: every community is labelled by its overlap with
domain seed sets, and the entanglement the domains genuinely share is
reported rather than hidden.

The unit stream and the markup screen are imported from ``collocation`` so
the two builds (#27 and #29) see exactly the same items.  Nothing is
repaired or re-tokenized here.

Reproduce::

    conda activate marker
    python work/cooccurrence_network.py
    python work/test_cooccurrence_network.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from collocation import load_bodies, load_markup, load_units, read_tsv, \
    write_tsv, cell

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "network"

SCHEMA = "network/1"
WINDOW = 5                      # sliding co-occurrence window (+/- units)
MIN_FREQ = 20                   # node floor: content-term frequency
MIN_COOCCUR = 3                 # edge floor: raw co-occurrence count
MIN_PMI = 1.0                   # edge floor: PMI >= 1.0 (2:1 over chance)
RESOLUTION = 1.0                # Louvain resolution
SEED = 42                       # Louvain seed (reproducibility)

# Non-content tokens that survive OCR as HTML entities or tag fragments.
EXTRA_STOP = {"lt", "gt", "amp", "quot"}

# Domain seed sets for the community-labelling pass.  A word may appear in
# several domains: the book fuses them (``embryo`` is body + genetics,
# ``blood`` is body + violence, ``clone`` is machine + genetics), and the
# labelling reports that overlap instead of forcing a single label.
DOMAIN_SEEDS = {
    "body": {"body", "organ", "brain", "flesh", "embryo", "placenta",
             "blood", "cell", "tissue", "heart", "nerve", "bone", "skin",
             "muscle", "entrails", "eyeball", "internal", "womb", "lobe",
             "joint", "masses"},
    "machine": {"machine", "tv", "hdd", "vtr", "load", "play", "rec",
                "upload", "download", "digital", "computer", "circuit",
                "screen", "monitor", "cyber", "hologram", "program",
                "code", "system", "device", "mechanism", "cyborg",
                "android", "replicant", "clone", "input", "output"},
    "violence": {"murder", "kill", "death", "fear", "scream", "torture",
                 "hunting", "assassin", "crime", "suicide", "rape",
                 "violence", "wound", "gun", "knife", "atrocity",
                 "murderous", "sutured", "defleshed", "hunted"},
    "sex": {"porn", "ecstasy", "love", "suck", "semen", "vagina", "porno",
            "erotic", "lust", "fuck", "cock", "breast", "orgasm",
            "sexual", "bondage", "rapes", "girl", "nude"},
    "genetics": {"gene", "dna", "chromosome", "embryo", "clone", "adam",
                 "boy-roid", "genome", "genetic", "mutation", "larva",
                 "retro-adam", "vital=serum", "tokage", "boy", "boys",
                 "feti", "retro"},
    "corporate": {"sato", "corporation", "company", "executive", "salary",
                  "business", "office", "manager", "president", "ceo",
                  "corp", "industry", "sato corporation"},
}


# ---------------------------------------------------------------------------
# stream construction (screened unit layer, function words removed)
# ---------------------------------------------------------------------------
def build_sequences():
    """Return (page_seqs, freq, n_items, screened_total, excluded_uids).

    page_seqs:      {page: [lower unit surface, ...]} in (page, start)
                    order, markup-screened and with function words removed.
    freq:           {lower surface: occurrence count} over the content stream.
    n_items:        content items (after function-word removal).
    screened_total: markup-screened items (before function-word removal).
    excluded_uids:  markup-residue unit uids removed by the screen.
    """
    page_units, _unit_meta = load_units()
    bodies = load_bodies()
    markup_spans, markup_uids, markup_types = load_markup(bodies)

    excluded = set()
    for page, units in page_units.items():
        spans = markup_spans.get(page, [])
        for s, uid, lower, surface, _cls in units:
            if uid in markup_uids or lower in markup_types:
                excluded.add(uid)
                continue
            if any(a <= s < b for a, b in spans):
                excluded.add(uid)

    stop = set(ENGLISH_STOP_WORDS) | EXTRA_STOP
    page_seqs = {}
    freq = collections.Counter()
    screened_total = 0
    for page in sorted(page_units):
        kept = [(uid, lower) for s, uid, lower, _surf, _cls
                in sorted(page_units[page]) if uid not in excluded]
        screened_total += len(kept)
        seq = [lower for uid, lower in kept if lower not in stop]
        page_seqs[page] = seq
        freq.update(seq)
    n_items = sum(len(s) for s in page_seqs.values())
    return page_seqs, freq, n_items, screened_total, excluded


# ---------------------------------------------------------------------------
# co-occurrence + PMI
# ---------------------------------------------------------------------------
def count_cooccurrence(page_seqs, window=WINDOW):
    """Sliding-window co-occurrence.

    Returns (cooccur, win_count, n_windows):
      cooccur   {(x, y): count}  unordered pairs, x < y, co-occurring in a
                            window (counted once per window position);
      win_count {term: count}   window positions containing the term;
      n_windows               total window positions (= screened items).
    """
    cooccur = collections.Counter()
    win_count = collections.Counter()
    for seq in page_seqs.values():
        L = len(seq)
        for i in range(L):
            lo = max(0, i - window)
            hi = min(L, i + window + 1)
            win = sorted(set(seq[lo:hi]))
            for x in win:
                win_count[x] += 1
            for a in range(len(win)):
                for b in range(a + 1, len(win)):
                    cooccur[(win[a], win[b])] += 1
    n_windows = sum(len(s) for s in page_seqs.values())
    return cooccur, win_count, n_windows


def pmi(f_xy, f_x, f_y, n):
    """PMI = log2(f_xy * N / (f_x * f_y)); 0 when any count is 0."""
    if f_xy <= 0 or f_x <= 0 or f_y <= 0 or n <= 0:
        return 0.0
    return math.log2((f_xy * n) / (f_x * f_y))


def build_graph(freq, cooccur, win_count, n_windows,
                min_freq=MIN_FREQ, min_cooccur=MIN_COOCCUR, min_pmi=MIN_PMI):
    """networkx Graph: nodes = terms (freq >= min_freq), edges = PMI >= min_pmi.

    Returns (G, edge_rows) where edge_rows carry the raw counts for the
    artifact table.
    """
    nodes = {t for t, c in freq.items() if c >= min_freq}
    G = nx.Graph()
    # sorted insertion: a set's iteration order varies with hash randomization,
    # and Louvain's result depends on the graph's node/edge ordering
    G.add_nodes_from(sorted(nodes))
    edge_rows = []
    for (x, y), f_xy in sorted(cooccur.items()):
        if f_xy < min_cooccur or x not in nodes or y not in nodes:
            continue
        p = pmi(f_xy, win_count[x], win_count[y], n_windows)
        if p < min_pmi:
            continue
        G.add_edge(x, y, weight=round(p, 4))
        edge_rows.append({"x": x, "y": y, "f_xy": f_xy,
                          "f_x": win_count[x], "f_y": win_count[y],
                          "n": n_windows, "pmi": round(p, 4)})
    edge_rows.sort(key=lambda r: (-r["pmi"], -r["f_xy"], r["x"], r["y"]))
    return G, edge_rows


# ---------------------------------------------------------------------------
# communities + centrality
# ---------------------------------------------------------------------------
def detect_communities(G, resolution=RESOLUTION, seed=SEED):
    """Louvain communities, largest first.  Returns a list of sets."""
    comms = nx.community.louvain_communities(
        G, weight="weight", resolution=resolution, seed=seed)
    comms.sort(key=len, reverse=True)
    return comms


def label_communities(comms, freq):
    """Label each community by its overlap with the domain seed sets.

    Returns {community_id: {"size", "top_terms", "domain", "domain_hits",
    "all_domains"}}.  ``domain`` is a descriptive label: the strongest
    domain, or ``dom1+dom2`` when the top two are close (the book fuses
    them), or ``mixed`` when no domain reaches two seed members.
    ``domain_hits`` counts seed members per domain, best first.
    """
    seed_to_domains = {}
    for dom, words in DOMAIN_SEEDS.items():
        for w in words:
            seed_to_domains.setdefault(w, set()).add(dom)
    out = {}
    for cid, members in enumerate(comms):
        hits = collections.Counter()
        for m in members:
            for dom in seed_to_domains.get(m, ()):
                hits[dom] += 1
        # Deterministic ranking: ties broken by domain name, not by
        # Counter/set insertion order (which is hash-randomized per process).
        ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
        if not ranked or ranked[0][1] < 2:
            domain = "mixed"
        elif len(ranked) > 1 and ranked[1][1] >= ranked[0][1] - 1:
            domain = f"{ranked[0][0]}+{ranked[1][0]}"
        else:
            domain = ranked[0][0]
        top = sorted(members, key=lambda t: (-freq.get(t, 0), t))
        out[cid] = {
            "size": len(members),
            "top_terms": top[:15],
            "domain": domain,
            "domain_hits": dict(ranked),
            "all_domains": [d for d, _ in ranked],
        }
    return out


def compute_centrality(G):
    """Five centrality measures for every node.

    Betweenness is unweighted (edge strength is a similarity, not a distance,
    so weighting it would invert the meaning); the other four use the PMI
    weight.  Returns {node: {measure: value}}.
    """
    deg = nx.degree_centrality(G)
    strength = dict(G.degree(weight="weight"))
    bet = nx.betweenness_centrality(G)
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eig = {n: 0.0 for n in G.nodes}
    pr = nx.pagerank(G, weight="weight")
    out = {}
    for n in G.nodes:
        out[n] = {
            "degree": round(deg.get(n, 0.0), 6),
            "strength": round(strength.get(n, 0.0), 4),
            "betweenness": round(bet.get(n, 0.0), 6),
            "eigenvector": round(eig.get(n, 0.0), 6),
            "pagerank": round(pr.get(n, 0.0), 6),
        }
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _md_table(rows, cols, headers, n=None):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in (rows if n is None else rows[:n]):
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return out


def write_report(path, G, comms, labels, cent, freq, edge_rows, diag):
    L = []
    A = L.append
    A("# Co-occurrence network and community detection (#29)")
    A("")
    A("Source: #17 unit layer (`units.tsv`), markup-screened with #19's "
      "ledger and the #27 tag-run detector, function words removed. "
      "Reproducible via `python work/cooccurrence_network.py`; regression "
      "suite `python work/test_cooccurrence_network.py`.")
    A("")
    A(f"Schema `{SCHEMA}`.  Nodes = content term types (frequency >= "
      f"{MIN_FREQ}); edges = PMI over a sliding ±{WINDOW}-unit window "
      f"(co-occurrence >= {MIN_COOCCUR}, PMI >= {MIN_PMI}); communities = "
      f"Louvain (resolution {RESOLUTION}, seed {SEED}).")
    A("")
    A("## 1. Headline result")
    A("")
    A(f"- **Graph**: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} "
      f"edges over {diag['content_items']:,} content-unit items "
      f"({diag['screened_items']:,} screened units minus "
      f"{diag['function_words_removed']:,} function-word occurrences).")
    A(f"- **Communities**: {len(comms)} Louvain communities, sizes "
      f"{[len(c) for c in comms]}.")
    top_bet = sorted(cent.items(), key=lambda kv: -kv[1]["betweenness"])[:10]
    A(f"- **Load-bearing concepts** (top betweenness): "
      + ", ".join(f"`{n}` ({v['betweenness']:.4f})" for n, v in top_bet))
    A("")
    A("The issue's hypothesis is that Louvain recovers the semantic domains "
      "body / machine / violence / sex / genetics / corporate.  §4 tests it "
      "community by community.")
    A("")
    A("## 2. Method")
    A("")
    A("- **Unit layer, never token indices.** The stream is #17's maximal "
      "units in `(page, start)` order; a page boundary can never join two "
      "items.  The markup screen (347 #19 residue occurrences + tag-furniture "
      "occurrences the general tag-run detector catches) is imported from "
      "`collocation`, so #27 and #29 see identical items.")
    A("- **Function words removed.** `sklearn`'s `ENGLISH_STOP_WORDS` plus "
      "the OCR HTML-entity residue `lt`/`gt`/`amp`/`quot`.  A network over "
      "function words measures the closed class, not the semantic domains; "
      "they are reported as removed, not silently dropped.")
    A(f"- **Sliding window ±{WINDOW} units.** Each unit position is one "
      "window; a pair co-occurs once per window position that contains "
      "both.  `N` = window positions = content items.")
    A(f"- **PMI** = `log2(f_xy * N / (f_x * f_y))`, the #27 formula.  Edges "
      f"need `f_xy >= {MIN_COOCCUR}` and `PMI >= {MIN_PMI}` (a 2:1 "
      "association over chance).")
    A(f"- **Louvain** (`networkx.community.louvain_communities`, weight = "
      f"PMI, resolution {RESOLUTION}, seed {SEED}).")
    A("- **Centrality**: degree, strength (weighted degree), betweenness "
      "(unweighted — PMI is a similarity, not a distance), eigenvector and "
      "PageRank (both PMI-weighted).")
    A("")
    A("## 3. The graph")
    A("")
    A(f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges.  Top "
      "edges by PMI:")
    A("")
    A("The strongest associations are the book's own fixed phrases and "
      "naming equations:")
    A("")
    L.extend(_md_table(edge_rows, ["x", "y", "f_xy", "pmi"],
                       ["x", "y", "f_xy", "PMI"], n=25))
    A("")
    A("## 4. Communities (Louvain)")
    A("")
    A("Each community is labelled by its overlap with domain seed sets "
      "(§6).  `top` lists the most frequent members.")
    A("")
    for cid, members in enumerate(comms):
        lab = labels[cid]
        top = sorted(members, key=lambda t: (-freq.get(t, 0), t))
        A(f"### C{cid} — {lab['domain']} (n={lab['size']})")
        A("")
        A("- top: " + ", ".join(
            f"`{t}` ({freq.get(t, 0)})" for t in lab["top_terms"][:15]))
        if lab["domain_hits"]:
            A("- domain hits: " + ", ".join(
                f"{d} {n}" for d, n in lab["domain_hits"].items()))
        cb = sorted((n for n in members if n in cent),
                    key=lambda n: -cent[n]["betweenness"])[:8]
        A("- highest-betweenness members: " + ", ".join(
            f"`{n}` ({cent[n]['betweenness']:.4f})" for n in cb))
        A("")
    A("## 5. Centrality — the load-bearing concepts")
    A("")
    A("Top 15 nodes by each measure.  Betweenness finds the bridges between "
      "communities; eigenvector/PageRank find the nodes embedded in strong "
      "neighbourhoods.")
    A("")
    A("**The betweenness leaders are not the most frequent words.** They are "
      "mid-frequency coinages and compounds — `sm` (the book's own abbreviation, "
      "as in `the SM drone`, `the SM disease`, `the SM circuit`), `clone-transmission`, "
      "`nano-machine`, `boyroid`, `lupus=space` — that appear across many "
      "different clause types and so sit on the shortest paths between "
      "communities.  That is the expected shape of a load-bearing concept in "
      "a fused vocabulary: the hubs are the coinages that *connect* domains, "
      "not the frequent English words that sit inside one.")
    A("")
    for measure, desc in (("betweenness", "betweenness (bridges)"),
                          ("eigenvector", "eigenvector"),
                          ("pagerank", "PageRank"),
                          ("strength", "strength (weighted degree)"),
                          ("degree", "degree")):
        ranked = sorted(cent.items(), key=lambda kv: -kv[1][measure])[:15]
        A(f"**{desc}**: " + ", ".join(
            f"`{n}` ({v[measure]:.4f})" for n, v in ranked))
        A("")
    A("## 6. Domain recovery assessment")
    A("")
    A("The issue names six domains: body / machine / violence / sex / "
      "genetics / corporate.  The table shows, for each community, the "
      "domains it draws seed words from (a community is labelled by its "
      "strongest domain; `mixed` means no domain reaches two seed members).")
    A("")
    A("| community | size | label | domain hits |")
    A("|---|---:|---|---|")
    for cid in range(len(comms)):
        lab = labels[cid]
        hits = ", ".join(f"{d} {n}" for d, n in lab["domain_hits"].items()) \
            or "—"
        A(f"| C{cid} | {lab['size']} | {lab['domain']} | {hits} |")
    A("")
    A("**Reading.** " + diag["domain_reading"])
    A("")
    A("**Verdict on the issue's hypothesis.** Louvain recovers *co-occurrence "
      "clusters*, not the six named domains as a clean partition.  The clusters "
      "are real and interpretable — C1 is the body/embryo block (`body`, "
      "`cadaver`, `flesh`, `embryo`, `organ`), C0 the ADAM/machine block "
      "(`adam`, `machine`, `memory`, `doll`, `mechanism`), C4 the "
      "machine/control block (`self`, `cyber`, `circuit`, `control`) — but "
      "they cut across the expected domain boundaries because Siratori's "
      "clauses fuse the domains (`internal organ` + `artificial sun` + `drug "
      "embryo` in one breath).  The two smallest expected domains are barely "
      "present at this frequency floor: `sex` survives only as scattered "
      "members (`rape`, `desire`, `ecstasy`, `bondage`) and `corporate` as "
      "`sato`/`corporation` in the 9-node C5.  This is a property of the "
      "text, not a failure of the method: a distributional network over a "
      "224-page coinage-dense book cannot separate domains that the author "
      "never separates.")
    A("")
    A("## 7. Method and limits")
    A("")
    A("- **PMI absolute value is normalization-dependent** (`N` = window "
      "positions); only rankings and the `PMI >= 1.0` threshold are "
      "interpreted.")
    A("- **A sliding window is an approximation.** It is a co-occurrence "
      "screen, not a dependency relation; a pair 5 units apart is treated "
      "like a pair 1 unit apart.")
    A("- **The graph is a sample of the lexicon.** Nodes need frequency >= "
      f"{MIN_FREQ}; rarer terms are invisible to community structure (they "
      "are still in #27's tables).")
    A("- **Communities are a hypothesis, not a ground truth.** The domain "
      "labels come from seed sets chosen from the book's own vocabulary; "
      "the entanglement the labels expose (body words co-occurring with "
      "machine words) is a property of the text, not a bug in the method.")
    A("- **Louvain is stochastic.** The seed is fixed (42) so the build "
      "reproduces exactly; a different seed can shift borderline nodes.")
    A("")
    A("## 8. Artifacts")
    A("")
    A("| file | contents |")
    A("|---|---|")
    A("| `graph_nodes.tsv` | node list: term, frequency, degree, community |")
    A("| `graph_edges.tsv` | edge list: x, y, f_xy, f_x, f_y, n, PMI |")
    A("| `communities.tsv` | community assignments: community, term, freq |")
    A("| `centrality.tsv` | five centrality measures per node |")
    A("| `graph.gexf` | the graph for Gephi / visualization |")
    A("| `network_report.md` | this report |")
    A("| `summary.json` | machine-readable headline numbers |")
    A("| `diagnostics.tsv` | build ledger |")
    A("| `reconciliation.tsv` | this build vs #17/#19/#27 |")
    A("")
    A("All `.tsv` are read with `split(\"\\t\")`, never a csv reader.")
    A("")
    A("---")
    A(f"*Generated by `work/cooccurrence_network.py` (schema `{SCHEMA}`).*")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="#29 co-occurrence network and community detection")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    ap.add_argument("--min-cooccur", type=int, default=MIN_COOCCUR)
    ap.add_argument("--min-pmi", type=float, default=MIN_PMI)
    ap.add_argument("--resolution", type=float, default=RESOLUTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    page_seqs, freq, n_items, screened_total, excluded = build_sequences()
    function_removed = screened_total - n_items

    cooccur, win_count, n_windows = count_cooccurrence(page_seqs, args.window)
    G, edge_rows = build_graph(
        freq, cooccur, win_count, n_windows,
        min_freq=args.min_freq, min_cooccur=args.min_cooccur,
        min_pmi=args.min_pmi)
    comms = detect_communities(G, resolution=args.resolution, seed=args.seed)
    labels = label_communities(comms, freq)
    cent = compute_centrality(G)

    # ---- domain reading sentence ----
    dom_counts = collections.Counter(
        labels[c]["domain"] for c in range(len(comms)))
    reading = (
        f"{len(comms)} communities: "
        + ", ".join(f"{d} {n}" for d, n in dom_counts.most_common())
        + ". The book's vocabulary is genuinely fused — body, machine and "
          "genetics terms co-occur because Siratori's clauses fuse them "
          "(`internal organ`, `artificial sun`, `drug embryo`) — so the "
          "communities are co-occurrence clusters that cut across the six "
          "named domains rather than a clean partition into them. The two "
          "smallest expected domains (`sex`, `corporate`) are barely present "
          "at this frequency floor.")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    diag = {
        "schema": SCHEMA,
        "generated": generated,
        "screened_items": screened_total,
        "markup_units_removed": len(excluded),
        "function_words_removed": function_removed,
        "content_items": n_items,
        "content_types": len(freq),
        "window": args.window,
        "min_freq": args.min_freq,
        "min_cooccur": args.min_cooccur,
        "min_pmi": args.min_pmi,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "communities": len(comms),
        "resolution": args.resolution,
        "seed": args.seed,
        "domain_reading": reading,
    }

    # ---- outputs ----
    node_rows = []
    for n in sorted(G.nodes):
        cid = next((i for i, c in enumerate(comms) if n in c), -1)
        node_rows.append({
            "term": n, "freq": freq.get(n, 0),
            "degree": G.degree(n), "community": cid,
        })
    write_tsv(out / "graph_nodes.tsv",
              ["term", "freq", "degree", "community"], node_rows)

    write_tsv(out / "graph_edges.tsv",
              ["x", "y", "f_xy", "f_x", "f_y", "n", "pmi"], edge_rows)

    comm_rows = []
    for cid, members in enumerate(comms):
        for m in sorted(members, key=lambda t: (-freq.get(t, 0), t)):
            comm_rows.append({"community": cid, "term": m,
                              "freq": freq.get(m, 0)})
    write_tsv(out / "communities.tsv", ["community", "term", "freq"], comm_rows)

    cent_rows = []
    for n in sorted(cent):
        cent_rows.append({"term": n, **cent[n]})
    write_tsv(out / "centrality.tsv",
              ["term", "degree", "strength", "betweenness", "eigenvector",
               "pagerank"], cent_rows)

    # GEXF with community + centrality as node attributes
    for n in G.nodes:
        cid = next((i for i, c in enumerate(comms) if n in c), -1)
        G.nodes[n]["community"] = cid
        G.nodes[n]["freq"] = freq.get(n, 0)
        for k, v in cent.get(n, {}).items():
            G.nodes[n][k] = v
    nx.write_gexf(G, str(out / "graph.gexf"))

    write_tsv(out / "diagnostics.tsv", ["diag", "value"],
              [{"diag": k, "value": v} for k, v in diag.items()])

    recon = [
        {"quantity": "units", "earlier": "74,665 (#17)",
         "here": screened_total + len(excluded),
         "note": "same #17 unit layer"},
        {"quantity": "screened occurrences", "earlier": "74,318 (#19/#24)",
         "here": screened_total,
         "note": f"minus {len(excluded)} markup occurrences "
                 f"(#19 ledger + tag-run detector), matching #27"},
        {"quantity": "content items", "earlier": "—",
         "here": n_items,
         "note": f"minus {function_removed} function-word occurrences "
                 f"(sklearn ENGLISH_STOP_WORDS + lt/gt/amp/quot)"},
        {"quantity": "collocation items", "earlier": "74,077 (#27)",
         "here": screened_total,
         "note": "the screened stream #27 scores bigrams over; #29 removes "
                 "function words before counting"},
    ]
    write_tsv(out / "reconciliation.tsv",
              ["quantity", "earlier", "here", "note"], recon)

    summary = {
        "schema": SCHEMA, "generated": generated,
        "counts": {k: v for k, v in diag.items() if k != "domain_reading"},
        "top_betweenness": [
            {"term": n, "betweenness": v["betweenness"]}
            for n, v in sorted(cent.items(),
                               key=lambda kv: -kv[1]["betweenness"])[:15]],
        "top_pmi_edges": edge_rows[:15],
        "community_sizes": [len(c) for c in comms],
        "community_labels": {str(c): labels[c]["domain"]
                             for c in range(len(comms))},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    write_report(out / "network_report.md", G, comms, labels, cent, freq,
                 edge_rows, diag)

    if not args.quiet:
        print(f"[network] nodes={G.number_of_nodes()} "
              f"edges={G.number_of_edges()} communities={len(comms)} "
              f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
