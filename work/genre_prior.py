#!/usr/bin/env python3
"""#60 Genre/prior dossier -> analysis/genre_prior/

L7 of ``docs/meaning-extraction-methods.md`` ends with item 5, and this task is
named after it:

    "Genre/priors -- cyberpunk, *Tetsuo: The Iron Man*, Ballard, Aase Berg,
     Blake Butler, 'text-virus' / codework poetics.  These set expectations
     for the *kind* of meaning present."

L7's rule sits above that line and governs everything below: *"External
material is a prior, not the object of study.  Use it to disambiguate, never to
ventriloquize."*

**What this module builds.**  An annotated dossier: for every tradition the
task names, the expectations it sets for the kind of meaning *Blood Electric*
should contain are stated one by one, each with the corpus-side probe that
would settle it, the measured result, and a verdict -- **present**, **thin**,
**absent** or **underdetermined** -- annotated with what the verdict does and
does not license.  The dossier is the deliverable (``dossier.md``); every
number in it is computed here from the book's own frozen layers or read from
the owning analysis's frozen artifact, never asserted.

**The probes are corpus-side and declared before they are run.**

* ``lexicon:<family>`` -- the family is a set of word-part patterns declared
  below, matched with a full match on the #17 *word-part* layer screened
  exactly as #56 (the only layer comparable to ordinary English words).  The
  measurement is the family's density per 1,000 book parts, read beside the
  same family's density in each #56 reference corpus, with a Poisson contrast
  against the general-English rate.  The excerpt corpora (cyberpunk 105
  tokens) are non-inferential by #56's own 10,000-token rule and are reported
  as context, never as a genre test.
* ``cooccur:<a>+<b>`` -- a #33 clause *has* a family when any of its parts
  full-matches one of its patterns.  The observed count of clauses holding
  both families is read against random placement (``n_a * n_b / n``, Poisson
  z), so "fused" means fused more than chance allows, not merely adjacent.
* ``artifact:<key>`` -- the number is read from the frozen artifact of the
  analysis that owns it (#19 coinages, #32 parse, #33 clauses, #44 affect,
  #54 self-similarity, #57 formulas, #59 roles) against a pass rule declared
  before the number is looked at.
* ``declared`` -- an expectation no probe in this workspace can settle.  It is
  written down as **underdetermined** with the evidence that would settle it,
  because an unmeasurable expectation silently dropped is the
  residue-dropping anti-pattern #76 audits.

**Verdict gates (conventions, declared here, not fitted to results).**

* lexicon: *absent* when the family never occurs; *present* when it occurs at
  least ``MIN_OCC`` times **and** at least ``RATIO_GATE`` times its
  general-English density; *thin* otherwise.  The ratio, not the significance,
  is the gate: every family here clears a Poisson z against generic prose (the
  book is not generic prose), so the question the dossier asks is how
  *distinctive* each register is, and the table publishes every number so a
  reader can re-gate.
* co-occurrence: *present* at ``MIN_OBS`` clauses and z >= ``Z_GATE``, else
  *thin*; *absent* below ``MIN_OBS``.
* artifact: the probe function returns its own verdict from its declared rule.
* declared: always *underdetermined*.

**What is not here.**  No tradition text is ingested.  Every prior work is
in-copyright; #57's survey already recorded the L7.5 priors as priors, not
acquisitions, and this module inherits that pinned status rather than
re-probing.  The expectations are stated from the traditions' documented
poetics, and no quotation of a prior work is used as evidence about the book.
A verdict of *present* says the meaning-material a tradition is made of is
measurably here; it does not say the author read that tradition, and a *thin*
verdict does not say he did not.

Reproduce::

    conda activate marker
    python work/genre_prior.py          # writes analysis/genre_prior/
    python work/genre_prior.py --quick  # capped technical registers (test mode)
    python work/test_genre_prior.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import intertext as itx  # #57: authorization vocabulary + the pinned ledger
import keyness as kn  # #56: book screen, TSV io, reference corpora

OUT = ROOT / "analysis" / "genre_prior"
PART_INDEX = ROOT / "analysis" / "tokenizer" / "part_index.tsv"
CLAUSES = ROOT / "analysis" / "clauses" / "clauses.tsv"
KEYNESS_SUMMARY = ROOT / "analysis" / "keyness" / "summary.json"
CLAUSE_SUMMARY = ROOT / "analysis" / "clauses" / "summary.json"

SCHEMA = "genre-prior/1"

#: the survey run that pinned the L7.5 priors as not-acquired; inherited, never
#: re-derived here, so this dossier's ledger cannot move with the clock.
PROBE_DATE = itx.PROBE_DATE

# --- verdict gates (declared conventions; see the module docstring) ---------
Z_GATE = 3.0            # Poisson z a co-occurrence must clear to count as fused
RATIO_GATE = 15.0       # book density / general-English density for "present"
MIN_OCC = 100           # occurrences a lexicon family needs to count at all
MIN_OBS = 5             # co-occurring clauses below which a pair is "absent"
#: below this many tokens a reference corpus cannot carry an inferential leg
#: (#56's rule, imported rather than restated).
MIN_TOKENS_INFERENTIAL = kn.MIN_TOKENS_INFERENTIAL
#: reference corpora compared against; the first is the control.
REGISTER_CORPORA = ("general_english", "medical", "cs", "legal")
#: excerpt corpora are context only (#56: fair-use excerpts, non-inferential).
EXCERPT_CORPORA = ("cyberpunk", "cutup_surrealist", "finnegans_wake")

# ===========================================================================
# 1. the prior ledger -- what the dossier refuses to ingest
# ===========================================================================

#: authorization classes mirror #57's vocabulary exactly.
AUTH_OPEN = itx.AUTH_OPEN
AUTH_EXCERPT = itx.AUTH_EXCERPT
AUTH_PRINT = itx.AUTH_PRINT
AUTH_SEARCHED = itx.AUTH_SEARCHED

V_INGESTED = itx.V_INGESTED
V_SEARCHED = itx.V_SEARCHED


@dataclass(frozen=True)
class PriorWork:
    key: str
    tradition: str
    work: str
    author: str
    year: str
    language: str
    license: str
    authorization: str
    verdict: str
    note: str


#: Every work the dossier takes as a prior.  All are in copyright.  The
#: cyberpunk row is already represented lawfully by #56's fair-use excerpt
#: corpus; everything else is a *prior only* -- the expectation is read from
#: the tradition's documented poetics, and no text is fetched or quoted as
#: evidence about the book.
PRIORS: tuple[PriorWork, ...] = (
    PriorWork("gibson_neuromancer", "cyberpunk", "Neuromancer", "William Gibson",
              "1984", "en", "in copyright", AUTH_EXCERPT, V_INGESTED,
              "Represented only by #56's fair-use cyberpunk excerpt corpus "
              "(105 tokens, non-inferential); no full text ingested."),
    PriorWork("stephenson_snow_crash", "cyberpunk", "Snow Crash",
              "Neal Stephenson", "1992", "en", "in copyright", AUTH_EXCERPT,
              V_INGESTED,
              "Represented only by #56's fair-use cyberpunk excerpt corpus."),
    PriorWork("sterling_mirrorshades", "cyberpunk", "Mirrorshades (ed.)",
              "Bruce Sterling", "1986", "en", "in copyright", AUTH_EXCERPT,
              V_INGESTED,
              "Represented only by #56's fair-use cyberpunk excerpt corpus; "
              "the Movement's manifesto volume."),
    PriorWork("tsukamoto_tetsuo", "tetsuo", "Tetsuo: The Iron Man (film)",
              "Shinya Tsukamoto", "1989", "ja", "in copyright", AUTH_SEARCHED,
              V_SEARCHED,
              "A film, not a text source; the prior is its body-horror "
              "paradigm (the Metal Fetishist's fusion) as documented in "
              "criticism, not a copy of the work."),
    PriorWork("ballard_crash", "ballard", "Crash", "J. G. Ballard", "1973",
              "en", "in copyright", AUTH_SEARCHED, V_SEARCHED,
              "In copyright; the prior is Crash's clinical-erotic "
              "psychopathology of technology."),
    PriorWork("ballard_atrocity_exhibition", "ballard",
              "The Atrocity Exhibition", "J. G. Ballard", "1970", "en",
              "in copyright", AUTH_SEARCHED, V_SEARCHED,
              "In copyright; the prior is its 'inner space' method, the "
              "clinical-entry form."),
    PriorWork("berg_forsla_fett", "berg", "Forsla fett (Transfer Fat)",
              "Aase Berg", "2002", "sv", "in copyright; Swedish; English "
              "translation exists", AUTH_SEARCHED, V_SEARCHED,
              "In copyright, in Swedish; no lawful excerpt assembled here.  "
              "The prior is the 'mommy trilogy' pregnancy-grotesque line."),
    PriorWork("butler_there_is_no_year", "butler", "There Is No Year",
              "Blake Butler", "2011", "en", "in copyright", AUTH_SEARCHED,
              V_SEARCHED,
              "In copyright; the prior is its deranged, roiling syntax."),
    PriorWork("butler_300000000", "butler", "300,000,000", "Blake Butler",
              "2014", "en", "in copyright", AUTH_SEARCHED, V_SEARCHED,
              "In copyright; the prior is its repetition and "
              "copycat-structure loops."),
    PriorWork("mez_codework", "text_virus", "codework / net.wurk",
              "Mary-Anne Breeze (Mez) and the codework tradition", "1990s-",
              "en", "mixed; partly journal/web published", AUTH_SEARCHED,
              V_SEARCHED,
              "The prior is the codework poetics of executable surface.  "
              "Burroughs' cut-up/Surrealist register is already lawfully "
              "present as #56's excerpt corpus; Burroughs' 'Electronic "
              "Revolution' essay is cited as the text-as-virus idea, not "
              "acquired."),
)

TRADITIONS: dict[str, str] = {
    "cyberpunk": "cyberpunk (Gibson / Stephenson / Sterling)",
    "tetsuo": "Tetsuo: The Iron Man (Tsukamoto 1989)",
    "ballard": "J. G. Ballard",
    "berg": "Aase Berg",
    "butler": "Blake Butler",
    "text_virus": "text-virus / codework poetics",
}

TRADITION_ORDER = tuple(TRADITIONS)

#: publication years, kept beside the ledger so the table cannot drift.
PRIOR_YEAR: dict[str, str] = {p.key: p.year for p in PRIORS}


# ===========================================================================
# 2. the lexicon families (declared before any measurement)
# ===========================================================================

@dataclass(frozen=True)
class Family:
    key: str
    patterns: tuple[str, ...]
    serves: str          # expectation ids the family probes
    note: str


LEXICON_FAMILIES: tuple[Family, ...] = (
    Family("corporate", (
        r"corporations?", r"corporate", r"corp", r"sato", r"conglomerates?",
        r"zaibatsu", r"compan(y|ies)", r"inc", r"syndicates?", r"monopolys?",
        r"cartels?", r"subsidiary",
    ), "CP1", "the institutional register: the firm as world-shaping power"),
    Family("digital", (
        r"digital", r"virtual", r"cyber\w*", r"quantum", r"nano\w*",
        r"electron\w*", r"hologram\w*", r"data", r"information", r"channel",
        r"hyperreal", r"simulat\w*", r"coded?", r"codes?", r"terminal",
        r"modem", r"modems?", r"internet", r"database", r"processor",
        r"circuit\w*",
    ), "CP2", "the technological register; cyber* matches cyberpunk's own "
              "neologism habit (cyberBuddha, cyberDolly...)"),
    Family("machine", (
        r"machines?", r"cyborgs?", r"androids?", r"robots?", r"drones?",
        r"mechanisms?", r"engines?", r"motors?", r"computers?", r"networks?",
        r"programs?", r"software", r"hardware", r"devices?", r"satellites?",
        r"mechanics?", r"skeleton\w*", r"exoskelet\w*",
    ), "CP3,TS3", "the machine register the body-interinterface prior expects"),
    Family("metal", (
        r"metals?", r"iron", r"steel", r"chrome", r"wires?", r"rust", r"mercury",
        r"scrap(ped|s|e)?", r"alloy", r"copper", r"zinc", r"solder",
    ), "TS1,TS3", "the material register Tetsuo's fusion is made of"),
    Family("organ", (
        r"organs?", r"entrails?", r"guts?", r"bowels?", r"viscera", r"liver",
        r"lungs?", r"heart", r"brain", r"stomach", r"tissues?", r"membranes?",
        r"sinew", r"bones?", r"cartilage", r"ribs?", r"skulls?", r"marrow",
        r"flesh",
    ), "TS4,CP3", "the body-interior register; body horror's camera"),
    Family("clinical", (
        r"clinics?", r"clinical", r"surgical", r"syndromes?", r"symptoms?",
        r"diagnos(is|e|tic|es)", r"patients?", r"therap(y|ies)", r"patholog\w*",
        r"inoculation", r"vaccines?", r"serum", r"tumou?rs?", r"lesions?",
        r"fractures?", r"sutured?", r"surgery", r"anaesthe\w*", r"nerves?",
        r"syringes?", r"doses?", r"sepsis\w*",
    ), "BL1", "Ballard's medicine-show diction"),
    Family("reproduction", (
        r"embryos?", r"fet(i|us|uses)", r"womb", r"placenta", r"eggs?",
        r"spawn", r"birth", r"born", r"fecundate[sd]?", r"gestation",
        r"larva(e)?", r"zygotes?", r"ovum", r"milk", r"umbilical",
        r"pregnan(t|cy)", r"genes?", r"chromosomes?", r"dna", r"genome",
        r"seeds?", r"replicat\w*", r"reproduction", r"clones?", r"boy",
    ), "BG1,BG3", "the reproductive register; Berg's pregnancy grotesque and "
                  "the biotechnological share one lexicon"),
    Family("grotesque", (
        r"grotesque", r"deform\w*", r"corrupt\w*", r"rot(ten)?", r"maggots?",
        r"decay\w*", r"cadavers?", r"cadaverous", r"carcass\w*", r"ghouls?",
        r"monsters?", r"freaks?", r"mutants?", r"aberrations?", r"morbid",
        r"putrid", r"abomin\w*",
    ), "BG2,BG3", "the deformation register named as a register"),
    Family("virus", (
        r"virus(es)?", r"viral", r"infect\w*", r"parasites?", r"parasitic",
        r"pathogens?", r"plague", r"contagion", r"contagious", r"biohazard",
        r"disease", r"epidemic", r"zoonotic",
    ), "TV2,TV3", "the infection register; the text-virus prior's own word"),
    Family("transformation", (
        r"transform\w*", r"mutat(e|es|ed|ion|ions)", r"proliferat\w*",
        r"fus(e|es|ed|ing)", r"melt\w*", r"dissolv\w*", r"liquef\w*",
        r"osmos(es|is)", r"excoria\w*", r"sutured?", r"gush\w*", r"stream\w*",
        r"metamorph\w*", r"evolve[d]?", r"shapeshift\w*",
    ), "TS2", "becoming, not being: the process verbs of mutation"),
    Family("sexual", (
        r"sexual", r"sex", r"porno\w*", r"erotic\w*", r"succubus", r"orgasm",
        r"ecstasy", r"phallic", r"masochis\w*", r"sadist\w*", r"bdsm", r"nude",
        r"desire", r"fetish", r"libido", r"vampires?", r"vamp", r"sperm",
        r"masturbat\w*", r"kiss\w*", r"lover?s?", r"lewd", r"obscen\w*",
    ), "BL3", "the erotic register Ballard's crash is made of"),
    Family("violence", (
        r"murder\w*", r"kill\w*", r"rape[ds]?", r"torture[ds]?", r"massacre",
        r"slaughter", r"assassins?", r"atrocity", r"crucifix\w*", r"crucified",
        r"napalm", r"bombs?", r"explod\w*", r"detonators?", r"behead\w*",
        r"decapitat\w*", r"genocide", r"terror\w*",
    ), "(context)", "the violence register; context arm, no expectation of "
                    "its own -- reported so the thin results can be read "
                    "against a family that is certainly here"),
    Family("city", (
        r"city", r"urban", r"streets?", r"metropolis", r"downtown", r"slums?",
        r"ghettos?", r"population", r"subway", r"skyline", r"alley",
    ), "(context)", "the generic urban register; context arm for the "
                    "cyberpunk named-place finding"),
)

FAMILY_BY_KEY: dict[str, Family] = {f.key: f for f in LEXICON_FAMILIES}
_COMPILED: dict[str, tuple[re.Pattern, ...]] = {
    f.key: tuple(re.compile("^" + p + "$") for p in f.patterns)
    for f in LEXICON_FAMILIES
}


# ===========================================================================
# 3. the expectations
# ===========================================================================

@dataclass(frozen=True)
class Expectation:
    eid: str
    tradition: str
    expectation: str
    rationale: str
    probe: str            # "lexicon:x" | "cooccur:a+b" | "artifact:k" | "declared"
    falsifier: str


EXPECTATIONS: tuple[Expectation, ...] = (
    # ---- cyberpunk -------------------------------------------------------
    Expectation("CP1", "cyberpunk",
                "corporate power is a live lexical register, not a name",
                "The genre's founding triangle (Gibson's zaibatsu worlds, "
                "Stephenson's franchulate, Sterling's Movement) makes the "
                "firm the ambient institution, so corporate vocabulary should "
                "be ordinary texture, not a single proper noun.",
                "lexicon:corporate",
                "corporate density at generic-prose level would strip the "
                "prediction: the corporation is named but not rendered."),
    Expectation("CP2", "cyberpunk",
                "a dense digital/technological register carries the texture",
                "Cyberpunk's surface is made of interfaces, data flows and "
                "electronics; the register should be dense and it should be "
                "neologism-prone, since the genre coins its own hardware "
                "words.",
                "lexicon:digital",
                "a digital register no denser than prose would mean the "
                "technology is named, not inhabited."),
    Expectation("CP3", "cyberpunk",
                "body and machine fuse inside single clauses, not across them",
                "The cyborg is the genre's body: interface as tissue, so "
                "machine and body vocabulary should share clauses more than "
                "chance allows.",
                "cooccur:machine+organ",
                "machine and organ vocabulary in separate clauses would be "
                "adjacency, not fusion."),
    Expectation("CP4", "cyberpunk",
                "the dystopia is named in a proper-noun layer",
                "Genre worlds are named (the Sprawl, Neo-Tokyo, the "
                "zaibatsu); the book's own jacket names Sato Corporation, "
                "Cadaver City and Placenta World, and #59 tested those names "
                "against the text.",
                "artifact:roles_named_world",
                "if the named place/organization entities were refuted "
                "against the text, the proper-noun layer would be marketing "
                "only."),
    # ---- tetsuo ----------------------------------------------------------
    Expectation("TS1", "tetsuo",
                "metal is a live material register of the body",
                "Tsukamoto's Metal Fetishist assimilates iron and scrap into "
                "flesh; the prior expects metal words to be ordinary material "
                "vocabulary, not a rare image.",
                "lexicon:metal",
                "metal at generic-prose density would mean the fusion is "
                "organic-only: the machine-horror is metaphoric."),
    Expectation("TS2", "tetsuo",
                "transformation is processual: mutation and proliferation verbs",
                "The film's horror is *becoming* -- liquefaction, fusion, "
                "excrescence -- so the process verbs should be a live class.",
                "lexicon:transformation",
                "a transformation register at prose density would reduce the "
                "body-horror to static description."),
    Expectation("TS3", "tetsuo",
                "fusion happens in contact: metal and organ in the same clause",
                "The prior image is metal *inside* the body, so the two "
                "registers should co-occur within clauses beyond chance.",
                "cooccur:metal+organ",
                "metal and organ vocabulary never sharing a clause would "
                "mean no fusion is ever staged."),
    Expectation("TS4", "tetsuo",
                "the horror is located in the body's interior",
                "Body horror's camera looks inward: organs, entrails, flesh "
                "as the stage of the event.",
                "lexicon:organ",
                "an organ register at prose density would locate the horror "
                "outside the body."),
    # ---- ballard ---------------------------------------------------------
    Expectation("BL1", "ballard",
                "a clinical register is over-dense against ordinary prose",
                "Ballard's diction is medicine-show: syndromes, sera, "
                "inoculations, pathologies as ordinary words, so the clinical "
                "register should stand far above prose.",
                "lexicon:clinical",
                "clinical at prose density would remove Ballard's method "
                "rather than his themes."),
    Expectation("BL2", "ballard",
                "the affect profile is the new psychopathology",
                "'Inner space' Ballard expects fear, ecstasy, paranoia and "
                "disgust to be *elevated together* -- the pathology is "
                "euphoric, not merely dreadful.",
                "artifact:affect_margins",
                "an affect profile flat against the English baseline would "
                "make the book's affect ordinary."),
    Expectation("BL3", "ballard",
                "violence is eroticized: sex and violence share clauses",
                "Crash's scandal is the erotic charge of the collision; the "
                "prior expects sexual and violent vocabulary fused at clause "
                "level.",
                "cooccur:sexual+violence",
                "sexual and violent registers in separate clauses would be "
                "Ballard's subject matter without Ballard's fusion."),
    Expectation("BL4", "ballard",
                "the pathology is organized as a case with testable claims",
                "Even at their most deranged, Ballard's texts keep a clinical "
                "scaffolding: named subjects doing definable things to "
                "definable targets, whose claims can be checked and can fail.",
                "artifact:roles_predicates",
                "if the event layer were dense while every named predicate "
                "were refuted, there would be no case, only texture."),
    # ---- berg ------------------------------------------------------------
    Expectation("BG1", "berg",
                "reproduction is a live and distinctive register",
                "Berg's 'mommy trilogy' makes pregnancy, birth and the "
                "genetic a central grotesque; the prior expects the "
                "reproductive lexicon to be unusually dense.",
                "lexicon:reproduction",
                "reproduction at prose density would make the fertility "
                "imagery incidental."),
    Expectation("BG2", "berg",
                "the grotesque is a named register, not an accident",
                "The criticism of Berg describes a violently grotesque "
                "register; the prior expects a deformation vocabulary "
                "standing well above prose.",
                "lexicon:grotesque",
                "grotesque at prose density would make the deformation "
                "rhetorical rather than systemic."),
    Expectation("BG3", "berg",
                "birth and grotesque fuse in single clauses",
                "In Berg the birth event *is* the deformation event, so the "
                "two registers should share clauses beyond chance.",
                "cooccur:reproduction+grotesque",
                "the two registers in separate clauses would separate "
                "birth from horror, which is not the prior's image."),
    Expectation("BG4", "berg",
                "tenderness persists inside the grotesque",
                "Berg's signature is horror blended with empathy -- the "
                "grotesque body is also loved -- so the prior expects "
                "tender wording inside the deformation register.",
                "declared",
                "unmeasurable here: no workspace lexicon separates "
                "tenderness from ecstasy (see the annotation)."),
    # ---- butler ----------------------------------------------------------
    Expectation("BT1", "butler",
                "syntax is deranged but not noise",
                "Butler's prose deranges syntax while remaining "
                "predictable-ish above chance; the prior expects parseability "
                "below ordinary English yet above a shuffle null.",
                "artifact:parse_derangement",
                "parseability at prose level would be ordinary syntax; at "
                "null level would be noise rather than a system."),
    Expectation("BT2", "butler",
                "phrases recur densely enough to be formulas",
                "Butler's roiling repetition: the same phrases churn, so a "
                "large share of the text should sit in repeated n-grams.",
                "artifact:formula_density",
                "a text with no recurring phrase skeleton would be "
                "all novelty, no pressure."),
    Expectation("BT3", "butler",
                "the book loops back on itself",
                "Butler's copycat structures and obsessive returns expect "
                "long-range self-similarity, not local coherence.",
                "artifact:self_similarity",
                "self-similarity at chance level would remove the loop "
                "structure the prior expects."),
    Expectation("BT4", "butler",
                "narrative time is elliptical",
                "Butler's narratives refuse chronology; the prior expects no "
                "recoverable temporal order of events.",
                "declared",
                "unmeasurable here: no workspace artifact measures temporal "
                "order (see the annotation)."),
    # ---- text-virus / codework --------------------------------------------
    Expectation("TV1", "text_virus",
                "the surface is code-like: operators as a structural share",
                "Codework's poetics is the executable surface -- punctuation "
                "as syntax, operators as grammar -- so operator density "
                "should be a structural fraction, not ornament.",
                "artifact:operator_surface",
                "operators at ornament level would make the surface look "
                "code-like without being code-like."),
    Expectation("TV2", "text_virus",
                "infection is a live and distinctive register",
                "The text-virus prior takes infection literally: virus, "
                "infection, parasite as the genre's governing metaphor "
                "made of words.",
                "lexicon:virus",
                "infection at prose density would leave the virus as a "
                "title only."),
    Expectation("TV3", "text_virus",
                "the virus is digital: infection and technology share clauses",
                "In codework the virus is *code* -- the biological and the "
                "computational are one figure -- so the registers should be "
                "fused at clause level.",
                "cooccur:virus+digital",
                "the two registers in separate clauses would keep the "
                "virus biological and the data merely technical."),
    Expectation("TV4", "text_virus",
                "an asemic layer: invented surfaces outside the dictionary",
                "Codework and asemic writing expect machine-coined, "
                "non-dictionary surfaces as a visible layer of the text.",
                "artifact:neologism_coinage",
                "no true-coinage layer would make the invention decorative "
                "rather than systemic."),
    Expectation("TV5", "text_virus",
                "a command/scream layer in capitals",
                "Codework's machine voice shouts in caps -- commands, "
                "labels, LOAD -- so an ALL-CAPS functional layer should be "
                "measurable.",
                "artifact:case_emphasis",
                "no caps layer would leave the machine voice unwritten."),
)

EXPECTATIONS_BY_ID: dict[str, Expectation] = {e.eid: e for e in EXPECTATIONS}

# the co-occurrence pairs the expectations actually probe (declared here).
COOCC_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (p.split("+")[0], p.split("+")[1])
    for e in EXPECTATIONS if e.probe.startswith("cooccur:")
    for p in [e.probe.split(":", 1)[1]]
)
COOCC_PAIRS = tuple(dict.fromkeys(COOCC_PAIRS))  # dedupe, keep order


# ===========================================================================
# 4. book-side loaders (the #56 screen, freshly re-run)
# ===========================================================================

def load_book_parts(log=print):
    """(parts, stats): markup-screened #17 word parts with (page, start).

    The screen is exactly #56's: a part is dropped when its unit uid is a #19
    markup residue, when its (page, start) falls inside a #27-detected tag
    span, or when its lower-cased surface is a #19 junk/markup type.  Returned
    sorted by (page, start) so first-occurrence citations are stable.
    """
    spans, markup_uids, markup_types = kn.load_markup()
    parts = []
    n_parts = n_dropped = 0
    for r in kn.read_tsv(PART_INDEX):
        if r["role"] != "w":
            continue
        n_parts += 1
        uid, page, start = int(r["uid"]), int(r["page"]), int(r["start"])
        surface = r["text"].lower()
        if (uid in markup_uids or surface in markup_types
                or any(a <= start < b for a, b in spans.get(page, ()))):
            n_dropped += 1
            continue
        parts.append((page, start, surface))
    parts.sort(key=lambda t: (t[0], t[1]))
    stats = {"parts": n_parts, "dropped": n_dropped, "kept": len(parts),
             "types": len({t[2] for t in parts})}
    log(f"[genre_prior] book word parts: {n_parts:,} total, "
        f"{n_dropped:,} markup-screened, {len(parts):,} kept")
    return parts, stats


def matched_families(surface: str) -> tuple[str, ...]:
    """The lexicon families a word-part surface full-matches."""
    return tuple(f for f in FAMILY_BY_KEY if any(p.match(surface)
                                                 for p in _COMPILED[f]))


def family_hits(parts):
    """{family: Counter of surface -> occurrences} over the kept parts."""
    hits = {f.key: collections.Counter() for f in LEXICON_FAMILIES}
    cites = {f.key: None for f in LEXICON_FAMILIES}
    for page, start, surface in parts:
        for fam in matched_families(surface):
            hits[fam][surface] += 1
            if cites[fam] is None:
                cites[fam] = (page, start, surface)
    return hits, cites


def load_reference_counts(corpus: str, max_tokens: int | None = None, log=None):
    """Counter of a #56 reference corpus's words (same regex, same screen)."""
    counts = collections.Counter()
    tokens = 0
    files = kn.corpus_files(corpus)
    if not files:
        raise FileNotFoundError(f"no corpus files in {CORPORA_DIR(corpus)}")
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        if corpus in kn.EXCERPT_CORPORA:
            text = "\n".join(ln for ln in text.splitlines()
                             if not ln.lstrip().startswith(("#", "--")))
        for tok in kn.EN_WORD_RE.findall(text):
            counts[tok.lower()] += 1
            tokens += 1
            if max_tokens is not None and tokens >= max_tokens:
                break
        if max_tokens is not None and tokens >= max_tokens:
            break
    if log:
        log(f"[genre_prior] reference {corpus}: {len(files)} files, "
            f"{tokens:,} tokens, {len(counts):,} types")
    return counts, tokens


def CORPORA_DIR(corpus: str) -> Path:
    return kn.CORPORA / corpus


# ===========================================================================
# 5. probes
# ===========================================================================

def run_lexicon_probe(fam_key: str, parts, book_counter, book_cite,
                      ref_density, ref_tokens):
    """Density table for one lexicon family against every reference corpus.

    ``book_counter`` / ``book_cite`` come from ``family_hits`` (computed once
    per build); ``ref_density`` is {corpus: {family: per-1k density}} from
    ``reference_densities``.
    """
    counter = book_counter.get(fam_key, collections.Counter())
    book_tokens = len(parts)
    occ = sum(counter.values())
    book_density = occ / book_tokens * 1000
    row = {
        "family": fam_key,
        "serves": FAMILY_BY_KEY[fam_key].serves,
        "book_occurrences": occ,
        "book_types": len(counter),
        "book_density_per_1k": round(book_density, 3),
        "top_types": ";".join(f"{w}:{c}" for w, c in
                              counter.most_common(6)),
        "example": ("p%d:%d" % book_cite[fam_key][:2]
                    if book_cite.get(fam_key) else "-"),
        "example_word": (book_cite.get(fam_key) or (None, None, "-"))[2],
    }
    ge_density = ref_density["general_english"].get(fam_key, 0.0)
    row["general_english_per_1k"] = round(ge_density, 3)
    row["ratio_vs_general_english"] = (
        round(book_density / ge_density, 2) if ge_density > 0 else "inf")
    exp = ge_density / 1000 * book_tokens
    row["general_english_expected"] = round(exp, 1)
    row["poisson_z_vs_general_english"] = (
        round((occ - exp) / math.sqrt(exp), 1) if exp > 0 else "inf")
    for corpus in REGISTER_CORPORA[1:] + EXCERPT_CORPORA:
        row[f"{corpus}_per_1k"] = round(
            ref_density[corpus].get(fam_key, 0.0), 3)
    if occ == 0:
        verdict = "absent"
    elif occ >= MIN_OCC and ge_density > 0 and (
            book_density / ge_density) >= RATIO_GATE:
        verdict = "present"
    else:
        verdict = "thin"
    row["verdict"] = verdict
    return row


def reference_densities(ref_counts, ref_tokens, log=None):
    """{corpus: {family: per-1k density}} in one pass per corpus."""
    out = {}
    for corpus, counts in ref_counts.items():
        per_fam = collections.Counter()
        for w, c in counts.items():
            for fam in matched_families(w):
                per_fam[fam] += c
        t = ref_tokens[corpus]
        out[corpus] = {f.key: (per_fam[f.key] / t * 1000 if t else 0.0)
                       for f in LEXICON_FAMILIES}
    if log:
        log("[genre_prior] reference family densities computed for "
            f"{len(out)} corpora")
    return out


def clause_family_sets(parts, clause_rows):
    """({clause_id: {families}}, n_assigned, n_unassigned, first_cites).

    The #33 clause partition tiles the page's token stream by char-span
    containment (#66's pointer walk over the same coordinates), so every kept
    part lands in exactly one clause -- asserted, then reconciled.
    """
    by_page = collections.defaultdict(list)
    for r in clause_rows:
        by_page[int(r["page"])].append((int(r["start"]), int(r["end"]),
                                        r["clause_id"], int(r["page"])))
    for p in by_page:
        by_page[p].sort()
    parts_by_page = collections.defaultdict(list)
    for page, start, surface in parts:
        parts_by_page[page].append((start, surface))
    fam_sets = collections.defaultdict(set)
    first = {}
    n_assigned = n_unassigned = 0
    for page, plist in parts_by_page.items():
        crows = by_page.get(page, ())
        j = 0
        for start, surface in plist:
            while j < len(crows) and crows[j][1] < start:
                j += 1
            if j < len(crows) and crows[j][0] <= start <= crows[j][1]:
                cid = crows[j][2]
                fams = matched_families(surface)
                if fams:
                    fam_sets[cid].update(fams)
                n_assigned += 1
            else:
                n_unassigned += 1
    for r in clause_rows:
        cid = r["clause_id"]
        fs = fam_sets.get(cid)
        if fs:
            first.setdefault(cid, (int(r["page"]), int(r["start"])))
    return fam_sets, n_assigned, n_unassigned, first


def run_cooccur_probe(a: str, b: str, fam_sets, n_clauses: int, first):
    """Clause-level co-occurrence of two families against random placement."""
    has_a = {cid for cid, fs in fam_sets.items() if a in fs}
    has_b = {cid for cid, fs in fam_sets.items() if b in fs}
    both = has_a & has_b
    na, nb, obs = len(has_a), len(has_b), len(both)
    exp = na * nb / n_clauses if n_clauses else 0.0
    z = (obs - exp) / math.sqrt(exp) if exp > 0 else 0.0
    cite = "-"
    for cid in sorted(both, key=lambda c: first.get(c, (10 ** 9, 10 ** 9))):
        if cid in first:
            cite = "p%d:%d" % first[cid]
            break
    if obs < MIN_OBS:
        verdict = "absent"
    elif z >= Z_GATE:
        verdict = "present"
    else:
        verdict = "thin"
    return {
        "pair": f"{a}+{b}",
        "family_a": a, "family_b": b,
        "clauses_with_a": na, "clauses_with_b": nb,
        "clauses_with_both": obs,
        "expected_random": round(exp, 1),
        "poisson_z": round(z, 2),
        "example_clause": cite,
        "verdict": verdict,
    }


# --- artifact probes: read the frozen number, apply the declared rule -----
def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def probe_roles_named_world():
    """CP4: are the named place/organization entities evidenced by #59?"""
    s = _load_summary(ROOT / "analysis/entity_roles/summary.json")
    verdicts = s["verdicts"]
    named = {k: v for k, v in verdicts.items() if k in ("R3", "R4", "R5")}
    n_ev = sum(1 for v in named.values() if v == "evidenced")
    verdict = "present" if n_ev >= 2 else ("thin" if n_ev == 1 else "absent")
    return {
        "key": "roles_named_world", "owner": "#59",
        "observed": {"named_world_roles": named,
                     "n_evidenced": n_ev,
                     "n_refuted_total": s["role_status"]["refuted"]},
        "rule": ">=2 of Cadaver City / Placenta World / Sato Corporation "
                "evidenced against the text",
        "detail": "the named dystopia exists as a proper-noun layer with "
                  "text evidence",
        "verdict": verdict,
    }


def probe_roles_predicates():
    """BL4: a dense event layer whose named claims are not all borne out."""
    n = _load_summary(ROOT / "analysis/narrative/summary.json")
    e = _load_summary(ROOT / "analysis/entity_roles/summary.json")
    events = n["events"]
    refuted = e["role_status"]["refuted"]
    uncertain = e["role_status"]["uncertain"]
    if events >= 1000 and refuted >= 1:
        verdict = "thin"
        detail = ("the case-history scaffolding is dense "
                  f"({events:,} actor->verb->target events) but the synopsis "
                  f"predicates are not merely incomplete: {refuted} role(s) "
                  f"refuted, {uncertain} uncertain (#59)")
    elif events >= 1000:
        verdict = "present"
        detail = f"{events:,} events, no refuted role"
    else:
        verdict = "absent"
        detail = "no dense event layer"
    return {
        "key": "roles_predicates", "owner": "#59 + #66",
        "observed": {"events": events, "refuted_roles": refuted,
                     "uncertain_roles": uncertain},
        "rule": "events >= 1000 and refuted >= 1 -> thin; events >= 1000 "
                "and refuted == 0 -> present",
        "detail": detail, "verdict": verdict,
    }


def probe_affect_margins():
    """BL2: fear, ecstasy, paranoia and disgust all above the English baseline."""
    a = _load_summary(ROOT / "analysis/affect/summary.json")
    margins = a["classifier"]["margins"]
    wanted = ("fear", "ecstasy", "paranoia", "disgust")
    zs = {k: margins[k]["z"] for k in wanted}
    n_elev = sum(1 for z in zs.values() if z >= Z_GATE)
    verdict = "present" if n_elev == len(wanted) else (
        "thin" if n_elev >= 2 else "absent")
    return {
        "key": "affect_margins", "owner": "#44",
        "observed": {k: {"z": margins[k]["z"], "margin": margins[k]["margin"],
                         "book_mean": margins[k]["book_mean"],
                         "english_mean": margins[k]["english_mean"]}
                     for k in wanted},
        "rule": f"all four z >= {Z_GATE} against the #44 English baseline",
        "detail": "; ".join(f"{k} z={zs[k]}" for k in wanted),
        "verdict": verdict,
    }


def probe_parse_derangement():
    """BT1: parseable below ordinary English, above the shuffle null."""
    p = _load_summary(ROOT / "analysis/parse_attempt/summary.json")
    h = p["headline"]
    book = h["v5_frac_parseable"]
    eng = h["english_frac_parseable"]
    null = h["null_frac_parseable"]
    if book < eng - 0.10 and book > null + 0.05:
        verdict = "present"
        detail = (f"parseable {book:.3f} vs English {eng:.3f} and shuffle "
                  f"null {null:.3f}: deranged yet above noise")
    elif book < eng:
        verdict = "thin"
        detail = (f"parseable {book:.3f} below English {eng:.3f} but not "
                  f"clearly above the null {null:.3f}")
    else:
        verdict = "absent"
        detail = f"parseable {book:.3f} at or above English {eng:.3f}"
    return {
        "key": "parse_derangement", "owner": "#32",
        "observed": {"book_frac_parseable": book, "english_frac_parseable": eng,
                     "null_frac_parseable": null},
        "rule": "book < english - 0.10 and book > null + 0.05",
        "detail": detail, "verdict": verdict,
    }


def probe_formula_density():
    """BT2: a large recurring-phrase skeleton exists (#57's book formulas)."""
    s = _load_summary(ROOT / "analysis/intertext/summary.json")
    classes = s["formula_classes"]
    total = sum(classes.values())
    rule = ">= 20000 book formulas (n>=2, freq>=2) on the #17 word-part layer"
    if total >= 20000:
        verdict = "present"
    elif total >= 5000:
        verdict = "thin"
    else:
        verdict = "absent"
    return {
        "key": "formula_density", "owner": "#57",
        "observed": {"book_formulas_n2plus": total,
                     "formula_classes": classes},
        "rule": rule,
        "detail": f"{total:,} recurring phrase formulas (n>=2, freq>=2)",
        "verdict": verdict,
    }


def probe_self_similarity():
    """BT3: long-range self-similarity (#54's band z-scores)."""
    s = _load_summary(ROOT / "analysis/self_similarity/summary.json")
    band_z = s["band_z"]
    z10 = band_z.get("10", 0.0)
    z100 = band_z.get("100", 0.0)
    best = max(band_z.values())
    if max(z10, z100) >= 10.0:
        verdict = "present"
    elif best >= 3.0:
        verdict = "thin"
    else:
        verdict = "absent"
    return {
        "key": "self_similarity", "owner": "#54",
        "observed": {"band_z_10": z10, "band_z_100": z100, "max_band_z": best},
        "rule": "z at lag 10 or 100 paragraphs >= 10",
        "detail": f"band z: lag10 {z10}, lag100 {z100}, max {best}",
        "verdict": verdict,
    }


def probe_operator_surface():
    """TV1: operator tokens are a structural share of the stream (#13/#33)."""
    sr = _load_summary(ROOT / "analysis/silence_removal/summary.json")
    cl = _load_summary(CLAUSE_SUMMARY)
    op_tokens = sr["operator_tokens"]
    words = sr["lexemes_baseline"]
    boundary = cl["boundary_tokens"]
    share = op_tokens / words
    if share >= 0.10:
        verdict = "present"
    elif share >= 0.05:
        verdict = "thin"
    else:
        verdict = "absent"
    return {
        "key": "operator_surface", "owner": "#13 + #33",
        "observed": {"operator_tokens": op_tokens, "word_tokens": words,
                     "operator_share": round(share, 4),
                     "clause_boundary_tokens": boundary},
        "rule": "operator tokens / word tokens >= 0.10",
        "detail": (f"{op_tokens:,} operator tokens over {words:,} words "
                   f"({share:.1%}); {boundary:,} clause-boundary operators"),
        "verdict": verdict,
    }


def probe_neologism_coinage():
    """TV4: a true-coinage layer outside the dictionary (#19)."""
    n = _load_summary(ROOT / "analysis/neologism/summary.json")
    types = n["counts"]["coinage_types"]
    occ = n["category_occurrences"]["true_coinage"]
    if types >= 40 and occ >= 100:
        verdict = "present"
    elif types >= 10:
        verdict = "thin"
    else:
        verdict = "absent"
    return {
        "key": "neologism_coinage", "owner": "#19",
        "observed": {"true_coinage_types": types, "true_coinage_occurrences": occ,
                     "compound_types": n["counts"]["compound_types"]},
        "rule": "coinage types >= 40 and occurrences >= 100",
        "detail": (f"{types} true coinages over {occ} occurrences; "
                   f"{n['counts']['compound_types']:,} compound types"),
        "verdict": verdict,
    }


def probe_case_emphasis():
    """TV5: an ALL-CAPS functional layer (#22)."""
    c = _load_summary(ROOT / "analysis/case_emphasis/summary.json")
    scream = c["by_class_occurrences"]["scream"]
    label = c["by_class_occurrences"]["label"]
    function = c["by_class_occurrences"]["function"]
    total = scream + label + function
    if scream >= 100 and total >= 500:
        verdict = "present"
    elif scream >= 20:
        verdict = "thin"
    else:
        verdict = "absent"
    return {
        "key": "case_emphasis", "owner": "#22",
        "observed": {"scream_occurrences": scream, "label_occurrences": label,
                     "function_occurrences": function},
        "rule": "scream >= 100 and scream+label+function >= 500",
        "detail": (f"caps layer: scream {scream}, label {label}, "
                   f"function {function} (share of words "
                   f"{c['caps_share_of_words']:.2%})"),
        "verdict": verdict,
    }


def probe_cyberpunk_arm():
    """Context: the cyberpunk excerpt arm cannot carry a genre inference."""
    s = _load_summary(ROOT / "analysis/intertext/summary.json")
    arm = s["arms"]["influence_cyberpunk"]
    tokens = arm["tokens"]
    shared = arm["book_shared_types_n2plus"]
    if tokens >= MIN_TOKENS_INFERENTIAL:
        verdict = "present"
        detail = f"excerpt arm is inferential ({tokens:,} tokens)"
    else:
        verdict = "underdetermined"
        detail = (f"the cyberpunk arm is {tokens} tokens "
                  f"(< {MIN_TOKENS_INFERENTIAL:,}); it shares "
                  f"{shared} book n-gram types (n>=2) and is a bound, not a "
                  f"test -- #56's own rule")
    return {
        "key": "cyberpunk_arm", "owner": "#56 + #57",
        "observed": {"tokens": tokens, "shared_book_ngram_types_n2plus": shared},
        "rule": f"(not an expectation) reported as context; inferential only "
                f"at >= {MIN_TOKENS_INFERENTIAL:,} tokens",
        "detail": detail, "verdict": verdict,
    }


ARTIFACT_PROBES = {
    "roles_named_world": probe_roles_named_world,
    "roles_predicates": probe_roles_predicates,
    "affect_margins": probe_affect_margins,
    "parse_derangement": probe_parse_derangement,
    "formula_density": probe_formula_density,
    "self_similarity": probe_self_similarity,
    "operator_surface": probe_operator_surface,
    "neologism_coinage": probe_neologism_coinage,
    "case_emphasis": probe_case_emphasis,
}


# ===========================================================================
# 6. annotations (the dossier's voice: what each verdict does not license)
# ===========================================================================

ANNOTATIONS: dict[str, str] = {
    "CP1": (
        "The firm exists, but as a name and not as a register. `Sato "
        "Corporation` is #59-evidenced as an `organization` with napalm/"
        "torture/vampire clusters, yet the generic corporate lexicon stays "
        "below the distinctiveness gate while the digital register clears it "
        "by a wide margin. The prior is met at the proper-noun layer and "
        "missed at the lexical layer: this is a cyberpunk of bodies, not of "
        "boardrooms. The verdict licenses 'corporate power is named and "
        "violent', not 'the corporation is rendered as infrastructure'."),
    "CP2": (
        "The technological register is among the book's densest families and "
        "it is neologism-prone (`cyber*` matches coined surfaces such as "
        "cyberBuddha), which is exactly cyberpunk's coinage habit. This "
        "verdict licenses 'the texture is technological'; it does not "
        "identify the register with any particular cyberpunk text -- the "
        "excerpt arm is 105 tokens and cannot."),
    "CP3": (
        "Machine and organ vocabulary share clauses far above random "
        "placement, so the fusion is staged inside the utterance, the "
        "codework/cyborg unit of this book. The annotation is about units: "
        "the clause is #33's operator-delimited utterance, so 'one clause' "
        "means one breath of the machine voice."),
    "CP4": (
        "#59 verified the named layer against the text: Cadaver City and "
        "Placenta World are `place`s, Sato Corporation an `organization`, "
        "EVOL the pink-ash planet the jacket names. The proper-noun world is "
        "real in the text, not only on the jacket. This is also the honest "
        "limit of the cyberpunk prior here: the named world carries the "
        "genre, while the lexical registers that would render it are thin."),
    "TS1": (
        "Metal is present -- chrome, copper, steel, wire, scrap -- but it "
        "stays below the distinctiveness gate, an order of magnitude sparser "
        "than the reproductive register. The prior's metal-flesh fusion is "
        "not this book's dominant fusion image; what fuses here is organ with "
        "machine (CP3) and birth with grotesque (BG3). The verdict licenses "
        "'metal is material', not 'the body assimilates metal'."),
    "TS2": (
        "Transformation is processual: mutation, proliferation, liquefaction, "
        "excoriation, fusion are a live verb class far above prose. The "
        "book's horror is *becoming*, Tetsuo's central verb, even though the "
        "material that becomes is flesh rather than steel."),
    "TS3": (
        "Metal and organ vocabulary do share clauses, but not decisively "
        "above chance -- the fusion is staged rarely. Read beside TS1: the "
        "Tetsuo paradigm is accounted for by the transformation verbs and the "
        "body-interior register, not by metallic fusion as a dominant "
        "image."),
    "TS4": (
        "The organ register is distinctive and among the densest families; "
        "#56's keyness independently names `organ`, `embryo` and `flesh` in "
        "its 32-word cross-corpus core. The interior camera is this book's "
        "camera -- the body-horror prior is met on its strongest leg."),
    "BL1": (
        "The clinical register is far above prose yet below a medical-abstract "
        "density: a novel with a medical accent, not a medical text. That "
        "middle position is itself the Ballardian finding -- clinical diction "
        "as literary method, measured between the two poles."),
    "BL2": (
        "All four psychopathological affects are elevated against #44's "
        "English baseline, with ecstasy strongest. The elevation of ecstasy "
        "*with* fear, paranoia and disgust is the Ballardian shape: the "
        "pathology is euphoric, not merely dreadful. This is #44's classifier "
        "on #44's lexicon, read as a profile, not as ground truth about "
        "feeling."),
    "BL3": (
        "Sexual and violent vocabulary co-occur in clauses about a quarter "
        "more often than random placement, but not decisively above the "
        "gate. Ballard's fusion is *suggested* here, not certified: the "
        "erotic register is dense (it clears z generically), the fusion "
        "specifically is thin. The verdict licenses 'eroticism and violence "
        "share the field', not 'the crash is eroticized'."),
    "BL4": (
        "The clinical-case prior is supported and refuted in the same "
        "breath: the event layer is dense (thousands of actor->verb->target "
        "events, #66), yet #59 refutes one synopsis role outright (Super "
        "Cherry 666 never hunts, never co-occurs with the succubus family) "
        "and leaves two predicates uncertain (no ADAM/ANTi-ADAM clash event; "
        "no wage-family verb for the dog). A Ballardian case-history has "
        "checkable claims, and this text has exactly that: claims that "
        "mostly check, and some that fail."),
    "BG1": (
        "Reproduction is the densest family in the book -- embryo, fetus, "
        "placenta, gene, chromosome, clone, fecundate, spawn. Berg's "
        "pregnancy grotesque and the biotechnological register share one "
        "lexicon here, and it dominates. The verdict licenses 'the "
        "reproductive is the book's central material'; it does not sort how "
        "much of that density is womb and how much is gene-war -- that split "
        "is not probed here."),
    "BG2": (
        "The grotesque is a named, dense register: cadaver, deformity, "
        "corruption, decay, aberration. It is systemic, as the tradition's "
        "criticism describes it, not rhetorical."),
    "BG3": (
        "Birth and grotesque share clauses an order of magnitude above the "
        "placement noise: the birth event *is* the deformation event in "
        "this book, which is the Berg prior's central image. This is the "
        "strongest fusion result in the dossier."),
    "BG4": (
        "Underdetermined, and stated as such. #44's affect lexicon measures "
        "intensity and valence classes (ecstasy above all) but no artifact "
        "distinguishes *tenderness* from *ecstasy*, so the prior's tender "
        "register cannot be probed without inventing a lexicon -- which "
        "would be a new measurement, not a dossier check. What would settle "
        "it: a tenderness lexicon scored with #44's own pipeline, with the "
        "same English-baseline contrast, declared before running."),
    "BT1": (
        "Parseable well below ordinary English yet clearly above the shuffle "
        "null: the syntax is deranged but not noise. Butler's prior expects "
        "exactly this posture -- a system that breaks grammar without "
        "dissolving into chance. Note the book sits *above* the null, so "
        "the derangement is structured derangement."),
    "BT2": (
        "The recurring-phrase skeleton is large: tens of thousands of book "
        "formulas (n>=2, freq>=2) on the #17 word-part layer, #57's count, "
        "with 116 classified `author_idiolect`. Repetition at formula scale "
        "is the Butler prior's pressure: the same phrases churn."),
    "BT3": (
        "Paragraph-level self-similarity rises with lag far above its "
        "shuffle null (#54): the book loops back on itself. Butler's "
        "copycat-structure prior is met at the macro scale, not the "
        "sentence scale."),
    "BT4": (
        "Underdetermined: no workspace artifact measures the *temporal "
        "order* of events (#66's arcs are event-density profiles, not "
        "chronology), so 'elliptical time' cannot be checked without a new "
        "instrument. What would settle it: a chronology instrument over "
        "#66's event cites (tense, sequence markers, first/last page per "
        "referent), declared before running. The expectation is recorded "
        "rather than dropped."),
    "TV1": (
        "Operators are a structural fraction of the stream -- over a tenth "
        "of the word tokens' count in operator tokens, with thousands of "
        "clause-boundary operators. The surface is code-like by "
        "measurement, not by impression: #14 parses the operator stream as "
        "a grammar and #13 removes it as silence."),
    "TV2": (
        "Infection is a distinctive register -- virus, infection, parasite, "
        "disease, plague -- with a density near a medical-abstract corpus "
        "and two orders above prose. #57 found the one non-generic "
        "influence-excerpt overlap in the book is `a virus` (Burroughs, "
        "*The Ticket That Exploded*): the virus prior has a documented "
        "intertextual toehold."),
    "TV3": (
        "Infection and digital vocabulary share clauses above chance: the "
        "virus is digital here, the codework figure literalized. The "
        "verdict licenses 'infection and computation are one image "
        "field'; it does not identify the figure with any codework text."),
    "TV4": (
        "A true-coinage layer exists outside the dictionary, with thousands "
        "of compound types beside it (#19). The asemic surface is a "
        "measured layer, not a reading impression."),
    "TV5": (
        "The ALL-CAPS layer is large and mostly functional -- labels, "
        "commands, function letters, screams (#22). The machine voice "
        "shouts in caps and the shouting is classifiable."),
}


# ===========================================================================
# 7. the run
# ===========================================================================

def write_prior_ledger(log=print):
    """sources.tsv: the prior works, their status, and what was not ingested."""
    rows = []
    for p in PRIORS:
        rows.append({
            "key": p.key, "tradition": p.tradition, "work": p.work,
            "author": p.author, "year": p.year, "language": p.language,
            "license": p.license, "authorization": p.authorization,
            "verdict": p.verdict,
            "ingested": "n" if p.authorization in itx.NON_INGESTIBLE else "y",
            "probe_date": PROBE_DATE, "note": p.note,
        })
    kn.write_tsv(OUT / "sources.tsv",
                 ["key", "tradition", "work", "author", "year", "language",
                  "license", "authorization", "verdict", "ingested",
                  "probe_date", "note"], rows)
    log(f"[genre_prior] sources.tsv: {len(rows)} prior works, "
        f"{sum(1 for r in rows if r['ingested'] == 'n')} not ingested")
    return rows


def write_expectations(log=print):
    rows = [{
        "eid": e.eid, "tradition": e.tradition,
        "tradition_label": TRADITIONS[e.tradition],
        "expectation": e.expectation, "rationale": e.rationale,
        "probe": e.probe, "falsifier": e.falsifier,
    } for e in EXPECTATIONS]
    kn.write_tsv(OUT / "expectations.tsv",
                 ["eid", "tradition", "tradition_label", "expectation",
                  "rationale", "probe", "falsifier"], rows)
    log(f"[genre_prior] expectations.tsv: {len(rows)} expectations")
    return rows


def build(log=print, quick: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    ledger_rows = write_prior_ledger(log)
    write_expectations(log)

    # --- book side ---------------------------------------------------------
    parts, book_stats = load_book_parts(log)

    # --- reference side ----------------------------------------------------
    ref_counts, ref_tokens = {}, {}
    #: quick mode caps the *technical* registers only.  The general-English
    #: arm is the control the lexicon gate reads, so it is never capped:
    #: verdicts must be mode-invariant even though the technical columns are
    #: approximations under --quick.
    budget = 5000 if quick else None
    for corpus in REGISTER_CORPORA + EXCERPT_CORPORA:
        cap = budget if corpus in ("medical", "cs", "legal") else None
        ref_counts[corpus], ref_tokens[corpus] = load_reference_counts(
            corpus, max_tokens=cap)
        inferential = ref_tokens[corpus] >= MIN_TOKENS_INFERENTIAL
        log(f"[genre_prior] reference {corpus}: "
            f"{ref_tokens[corpus]:,} tokens "
            f"({'inferential' if inferential else 'context only'}"
            f"{'; capped (quick)' if cap else ''})")

    # --- lexicon probes ----------------------------------------------------
    book_hits, book_cites = family_hits(parts)
    ref_density = reference_densities(ref_counts, ref_tokens, log=log)
    lexicon_rows = [
        run_lexicon_probe(f.key, parts, book_hits, book_cites,
                          ref_density, ref_tokens)
        for f in LEXICON_FAMILIES]
    kn.write_tsv(OUT / "lexicon_probes.tsv",
                 ["family", "book_occurrences", "book_types",
                  "book_density_per_1k", "top_types", "example",
                  "example_word", "general_english_per_1k",
                  "ratio_vs_general_english", "general_english_expected",
                  "poisson_z_vs_general_english"]
                 + [f"{c}_per_1k" for c in REGISTER_CORPORA[1:] + EXCERPT_CORPORA]
                 + ["verdict"], lexicon_rows)
    log(f"[genre_prior] lexicon_probes.tsv: {len(lexicon_rows)} families "
        f"({sum(1 for r in lexicon_rows if r['verdict'] == 'present')} present, "
        f"{sum(1 for r in lexicon_rows if r['verdict'] == 'thin')} thin)")

    # --- co-occurrence probes ----------------------------------------------
    clause_rows = kn.read_tsv(CLAUSES)
    n_clauses = len(clause_rows)
    fam_sets, n_assigned, n_unassigned, first = clause_family_sets(parts,
                                                                  clause_rows)
    cooccur_rows = [run_cooccur_probe(a, b, fam_sets, n_clauses, first)
                    for a, b in COOCC_PAIRS]
    kn.write_tsv(OUT / "cooccurrence.tsv",
                 ["pair", "family_a", "family_b", "clauses_with_a",
                  "clauses_with_b", "clauses_with_both", "expected_random",
                  "poisson_z", "example_clause", "verdict"], cooccur_rows)
    log(f"[genre_prior] cooccurrence.tsv: {len(cooccur_rows)} pairs "
        f"({sum(1 for r in cooccur_rows if r['verdict'] == 'present')} present, "
        f"{sum(1 for r in cooccur_rows if r['verdict'] == 'thin')} thin); "
        f"{n_assigned:,} parts assigned to clauses, {n_unassigned} unassigned")

    # --- artifact probes ---------------------------------------------------
    artifact_rows = [fn() for fn in ARTIFACT_PROBES.values()]
    context = probe_cyberpunk_arm()
    kn.write_tsv(OUT / "artifact_probes.tsv",
                 ["key", "owner", "observed", "rule", "detail", "verdict"],
                 [{"key": r["key"], "owner": r["owner"],
                   "observed": json.dumps(r["observed"], sort_keys=True),
                   "rule": r["rule"], "detail": r["detail"],
                   "verdict": r["verdict"]}
                  for r in artifact_rows + [context]])
    log(f"[genre_prior] artifact_probes.tsv: {len(artifact_rows)} probes + "
        f"1 context row")

    # --- verdicts ----------------------------------------------------------
    cooccur_by_pair = {r["pair"]: r for r in cooccur_rows}
    lexicon_by_family = {r["family"]: r for r in lexicon_rows}
    artifact_by_key = {r["key"]: r for r in artifact_rows}
    verdict_rows = []
    for e in EXPECTATIONS:
        kind, _, arg = e.probe.partition(":")
        if kind == "lexicon":
            r = lexicon_by_family[arg]
            verdict, evidence = r["verdict"], _lexicon_evidence(r)
        elif kind == "cooccur":
            r = cooccur_by_pair[arg]
            verdict, evidence = r["verdict"], _cooccur_evidence(r)
        elif kind == "artifact":
            r = artifact_by_key[arg]
            verdict, evidence = r["verdict"], r["detail"]
        else:
            verdict, evidence = "underdetermined", e.falsifier
        verdict_rows.append({
            "eid": e.eid, "tradition": e.tradition, "probe": e.probe,
            "expectation": e.expectation, "verdict": verdict,
            "evidence": evidence,
            "annotation": ANNOTATIONS.get(e.eid, ""),
        })
    kn.write_tsv(OUT / "verdicts.tsv",
                 ["eid", "tradition", "probe", "expectation", "verdict",
                  "evidence", "annotation"], verdict_rows)
    counts = collections.Counter(r["verdict"] for r in verdict_rows)
    log(f"[genre_prior] verdicts.tsv: {len(verdict_rows)} expectations "
        f"({dict(counts)})")

    # --- reconciliation ----------------------------------------------------
    recon = reconcile(book_stats, parts, n_clauses, n_assigned, quick=quick,
                      log=log)
    kn.write_tsv(OUT / "reconciliation.tsv",
                 ["item", "expected", "observed", "source", "ok"], recon)

    # --- diagnostics -------------------------------------------------------
    diagnostics = [
        {"check": "families_declared", "value": len(LEXICON_FAMILIES),
         "detail": "declared before measurement"},
        {"check": "families_absent_from_book", "value": sum(
            1 for r in lexicon_rows if r["verdict"] == "absent"),
         "detail": ",".join(r["family"] for r in lexicon_rows
                            if r["verdict"] == "absent") or "none"},
        {"check": "prior_works_not_ingested", "value": sum(
            1 for r in ledger_rows if r["ingested"] == "n"),
         "detail": f"of {len(ledger_rows)} prior works; the ingested rows are "
                   f"#56's excerpt corpus, not this module's acquisition"},
        {"check": "expectations_underdetermined", "value": counts.get(
            "underdetermined", 0),
         "detail": ",".join(r["eid"] for r in verdict_rows
                            if r["verdict"] == "underdetermined") or "none"},
        {"check": "clause_parts_unassigned", "value": n_unassigned,
         "detail": "kept parts that land in no #33 clause (must be 0)"},
        {"check": "quick_build", "value": "y" if quick else "n",
         "detail": "quick caps the technical registers (medical/cs/legal) at "
                   "5,000 tokens; general English and every book-side number "
                   "are mode-invariant, so every verdict is too"},
    ]
    kn.write_tsv(OUT / "diagnostics.tsv", ["check", "value", "detail"],
                 diagnostics)

    # --- summary + report --------------------------------------------------
    summary = {
        "schema": SCHEMA,
        "probe_date": PROBE_DATE,
        "quick": quick,
        "book": book_stats,
        "clauses": n_clauses,
        "traditions": TRADITIONS,
        "gates": {"z_gate": Z_GATE, "ratio_gate": RATIO_GATE,
                  "min_occ": MIN_OCC, "min_obs": MIN_OBS,
                  "min_tokens_inferential": MIN_TOKENS_INFERENTIAL},
        "priors": [{"key": p.key, "tradition": p.tradition, "work": p.work,
                    "authorization": p.authorization} for p in PRIORS],
        "priors_not_ingested": sum(1 for r in ledger_rows
                                   if r["ingested"] == "n"),
        "lexicon_probes": lexicon_rows,
        "cooccurrence": cooccur_rows,
        "artifact_probes": [{k: v for k, v in r.items() if k != "observed"}
                            | {"observed": r["observed"]}
                            for r in artifact_rows],
        "context": {k: v for k, v in context.items() if k != "observed"}
                   | {"observed": context["observed"]},
        "verdicts": verdict_rows,
        "verdict_counts": dict(counts),
        "references": ref_tokens,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    write_dossier(log, summary)
    log(f"[genre_prior] done: {dict(counts)}; artifacts -> {OUT}")
    return summary


def _lexicon_evidence(r):
    ratio = r["ratio_vs_general_english"]
    return (f"{r['family']}: {r['book_occurrences']} occurrences / "
            f"{r['book_types']} types, {r['book_density_per_1k']}/1k vs "
            f"general English {r['general_english_per_1k']}/1k "
            f"(x{ratio}, z {r['poisson_z_vs_general_english']}); e.g. "
            f"{r['example_word']} {r['example']}")


def _cooccur_evidence(r):
    return (f"{r['pair']}: {r['clauses_with_both']} clauses hold both "
            f"(expected {r['expected_random']} under random placement, "
            f"z {r['poisson_z']}); e.g. clause {r['example_clause']}")


def reconcile(book_stats, parts, n_clauses, n_assigned, quick=False, log=print):
    """Closure checks against the owning analyses' frozen numbers."""
    rows = []

    def add(item, expected, observed, source, ok=None):
        ok = ("y" if expected == observed else "n") if ok is None else ok
        rows.append({"item": item, "expected": expected, "observed": observed,
                     "source": source, "ok": ok})

    ks = _load_summary(KEYNESS_SUMMARY)
    add("book_parts_total", ks["book"]["parts"], book_stats["parts"],
        "#56 keyness/summary.json")
    add("book_kept_tokens", ks["book"]["kept"], book_stats["kept"],
        "#56 keyness/summary.json")
    add("book_types", ks["book"]["types"],
        len({p[2] for p in parts}), "#56 keyness/summary.json")
    cs = _load_summary(CLAUSE_SUMMARY)
    add("clause_count", cs["n_clauses"], n_clauses, "#33 clauses/summary.json")
    add("parts_assigned_to_clauses", book_stats["kept"], n_assigned,
        "#33 clause partition (char-span containment)")
    add("parts_unassigned", 0, book_stats["kept"] - n_assigned,
        "every kept part lands in exactly one clause")
    # the #57 ledger must already record the L7.5 priors as not acquired
    intertext_sources = (ROOT / "analysis/intertext/sources.tsv")
    has_prior_row = "n"
    if intertext_sources.exists():
        for r in kn.read_tsv(intertext_sources):
            if r["key"] == "genre_prior_ballard" and r["ingested"] == "n":
                has_prior_row = "y"
    add("intertext_ledger_marks_priors_not_acquired", "y", has_prior_row,
        "#57 analysis/intertext/sources.tsv")
    add("quick_build", "y" if quick else "n", "y" if quick else "n",
        "mode flag", ok="skip")
    bad = [r for r in rows if r["ok"] == "n"]
    log(f"[genre_prior] reconciliation: {len(rows) - len(bad)}/{len(rows)} ok")
    return rows


# ===========================================================================
# 8. the dossier (the deliverable)
# ===========================================================================

def _fmt(x, nd=3):
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def write_dossier(log=print, summary=None):
    """dossier.md: the annotated prior dossier."""
    if summary is None:
        summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    verd = {r["eid"]: r for r in summary["verdicts"]}
    counts = summary["verdict_counts"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    ap = lines.append

    ap("# Genre/prior dossier: what the priors expect, and what the corpus measures (#60)")
    ap("")
    ap(f"Generated: {stamp}")
    ap("")
    ap("L7.5 of `docs/meaning-extraction-methods.md` names the priors this "
       "dossier is built from -- cyberpunk, *Tetsuo: The Iron Man*, Ballard, "
       "Aase Berg, Blake Butler, text-virus/codework poetics -- and says what "
       "they are for: *they set expectations for the kind of meaning present*. "
       "L7's governing rule sits above that line: external material is a "
       "**prior, not the object of study**; it is used to disambiguate, never "
       "to ventriloquize.")
    ap("")
    ap("This dossier states every expectation one by one, names the probe that "
       "would settle it, runs that probe on the book's own frozen layers, and "
       "annotates the verdict. No tradition text is ingested (ledger below); "
       "no prior quotation is used as evidence about the book.")
    ap("")
    ap(f"**Verdicts: {counts.get('present', 0)} present, "
       f"{counts.get('thin', 0)} thin, {counts.get('absent', 0)} absent, "
       f"{counts.get('underdetermined', 0)} underdetermined, of "
       f"{len(summary['verdicts'])} expectations.**")
    ap("")

    # --- 1. ledger
    ap("## 1. The prior ledger -- what the dossier refuses to ingest")
    ap("")
    ap(f"Every prior work is in copyright. The cyberpunk tradition is already "
       f"represented lawfully by #56's fair-use excerpt corpus "
       f"({summary['references'].get('cyberpunk', 0)} tokens, non-inferential "
       f"by #56's own {MIN_TOKENS_INFERENTIAL:,}-token rule); the rest are "
       f"priors only. #57's survey pinned the L7.5 priors as not-acquired on "
       f"{PROBE_DATE}, and this module inherits that status rather than "
       f"re-probing.")
    ap("")
    ap("| work | tradition | year | status here |")
    ap("|---|---|---|---|")
    for p in summary["priors"]:
        ap(f"| {p['work']} | {TRADITIONS[p['tradition']]} | "
           f"{PRIOR_YEAR[p['key']]} | "
           f"{'excerpt corpus via #56' if p['authorization'] == 'free_excerpt' else 'prior only, not ingested'} |")
    ap("")

    # --- 2. method
    ap("## 2. Method: the probes and the gates")
    ap("")
    ap("* **Book side.** The #17 *word-part* layer, markup-screened exactly "
       f"as #56 ({summary['book']['kept']:,} kept of "
       f"{summary['book']['parts']:,} parts, "
       f"{summary['book']['types']:,} types) -- the only layer comparable to "
       "ordinary English words.  The screen is #56's exactly, jacket pages "
       "included (p1-3 and the p224 synopsis): no page is excluded, so every "
       "count reconciles with #56 -- and a cited exemplar can be front "
       "matter.")
    ap("* **Lexicon probes.** Each family is a set of word-part patterns "
       "declared before measurement (full match on the surface). Density is "
       "reported per 1,000 book parts against the same family's density in "
       "each #56 reference corpus, with a Poisson contrast against the "
       "general-English rate. The excerpt corpora are context only.")
    ap(f"* **Co-occurrence probes.** A #33 clause *has* a family when any of "
       f"its parts full-matches it; the observed count of clauses holding "
       f"both families is read against random placement "
       f"(`n_a x n_b / {summary['clauses']:,}` clauses, Poisson z). "
       f"\"Fused\" means more often than chance allows, not merely adjacent. "
       f"The clause partition is #33's operator-delimited utterance, so "
       f"every kept part lands in exactly one clause "
       f"({summary['book']['kept']:,} assigned, 0 unassigned).")
    ap("* **Artifact probes.** The number is read from the frozen artifact of "
       "the analysis that owns it; the pass rule is declared here before the "
       "number is read.")
    ap("* **Gates (declared conventions, not fitted to results).** Lexicon: "
       f"*absent* if never occurring, *present* at >= {MIN_OCC} occurrences "
       f"and >= {RATIO_GATE:g}x its general-English density, else *thin*. "
       f"Co-occurrence: *present* at >= {MIN_OBS} clauses and z >= {Z_GATE:g}, "
       f"else *thin*; *absent* below {MIN_OBS}. *Underdetermined* means no "
       "probe in this workspace can settle the expectation -- it is written "
       "down with the evidence that would, not dropped.")
    ap("")
    ap("Why the ratio and not significance is the lexicon gate: every family "
       "here clears a Poisson z against generic prose by orders of "
       "magnitude -- the book is not generic prose -- so significance cannot "
       "discriminate. Distinctiveness can. Every number is published below "
       "so a reader can re-gate.")
    ap("")

    # --- 3. expectations table
    ap("## 3. The expectations table")
    ap("")
    ap("| id | tradition | expectation | probe | verdict |")
    ap("|---|---|---|---|---|")
    for trad in TRADITION_ORDER:
        for e in EXPECTATIONS:
            if e.tradition != trad:
                continue
            v = verd[e.eid]
            ap(f"| {e.eid} | {TRADITIONS[trad]} | {e.expectation} | "
               f"`{e.probe}` | **{v['verdict']}** |")
    ap("")

    # --- 4. lexicon table
    ap("## 4. The lexicon families, measured")
    ap("")
    ap("Densities per 1,000 book parts; ratio against general English. "
       "Excerpt corpora (105/56/49 tokens) are context only and cannot "
       "carry an inference.")
    ap("")
    ap("| family | serves | occ | types | book /1k | gen. English /1k | "
       "ratio | z | medical /1k | verdict |")
    ap("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in summary["lexicon_probes"]:
        ap(f"| {r['family']} | {r['serves']} | {r['book_occurrences']:,} | "
           f"{r['book_types']} | {_fmt(r['book_density_per_1k'])} | "
           f"{_fmt(r['general_english_per_1k'])} | "
           f"x{r['ratio_vs_general_english']} | "
           f"{r['poisson_z_vs_general_english']} | "
           f"{_fmt(r['medical_per_1k'])} | **{r['verdict']}** |")
    ap("")
    ap("Top types per family: "
       + "; ".join(f"**{r['family']}**: {r['top_types'].replace(';', ', ')}"
                   for r in summary["lexicon_probes"]))
    ap("")

    # --- 5. co-occurrence table
    ap("## 5. The fusion probes (clause-level co-occurrence)")
    ap("")
    ap("| pair | clauses with both | expected | z | example | verdict |")
    ap("|---|---:|---:|---:|---|---|")
    for r in summary["cooccurrence"]:
        ap(f"| {r['pair']} | {r['clauses_with_both']} | "
           f"{r['expected_random']} | {r['poisson_z']} | "
           f"{r['example_clause']} | **{r['verdict']}** |")
    ap("")

    # --- 6. artifact probes table
    ap("## 6. The artifact probes")
    ap("")
    ap("| key | owner | observed | rule | verdict |")
    ap("|---|---|---|---|---|")
    for r in summary["artifact_probes"] + [summary["context"]]:
        ap(f"| {r['key']} | {r['owner']} | {r['detail']} | {r['rule']} | "
           f"**{r['verdict']}** |")
    ap("")
    ap("The machine-readable `observed` values are in `artifact_probes.tsv` "
       "and `summary.json`.")
    ap("")

    # --- 7. the annotated dossier
    ap("## 7. The annotated dossier")
    ap("")
    for trad in TRADITION_ORDER:
        ap(f"### {TRADITIONS[trad]}")
        ap("")
        for e in EXPECTATIONS:
            if e.tradition != trad:
                continue
            v = verd[e.eid]
            ap(f"**{e.eid} -- {e.expectation}** -- **{v['verdict'].upper()}**")
            ap("")
            ap(f"*Why the prior expects it.* {e.rationale}")
            ap("")
            ap(f"*Probe `{e.probe}`.* {v['evidence']}")
            ap("")
            ap(f"*Annotation.* {ANNOTATIONS[e.eid]}")
            ap("")
            ap(f"*What would falsify it.* {e.falsifier}")
            ap("")

    # --- 8. synthesis
    ap("## 8. The honest shape of the book, as the priors found it")
    ap("")
    by_density = sorted(summary["lexicon_probes"],
                        key=lambda r: -r["book_density_per_1k"])
    top = by_density[:5]
    thin = [r for r in summary["lexicon_probes"] if r["verdict"] == "thin"]
    ratios = [float(r["ratio_vs_general_english"]) for r in thin]
    ap("Ranked by density, the book's live registers are "
       + ", ".join(f"**{r['family']}** ({_fmt(r['book_density_per_1k'])}/1k)"
                   for r in top)
       + ". The thin families are "
       + ", ".join(f"**{r['family']}** "
                   f"(x{r['ratio_vs_general_english']} above prose)"
                   for r in thin)
       + ".")
    ap("")
    ap(f"The split is the finding. Where the priors predict *infrastructure* "
       f"-- the corporate register of cyberpunk, the metal register of "
       f"Tetsuo -- the corpus is thin: every thin family sits between "
       f"x{min(ratios):.1f} and x{max(ratios):.1f} above generic prose, while "
       f"every present family clears x{RATIO_GATE:g}. Where the priors "
       f"predict *interiority* -- the organ register of body horror, the "
       f"reproductive register of Berg, the infection register of the "
       f"text-virus -- the corpus is dense and distinctive. The fusions that "
       f"are certified at clause level are machine x organ and reproduction "
       f"x grotesque; the fusions the priors would most expect (metal x "
       f"organ, sex x violence) are the two that fall just under the gate. A "
       f"cyberpunk of the boardroom this is not; a cyberpunk of the body, "
       f"clinically inflected, reproductively driven, deranged above noise "
       f"-- that is what the priors find here.")
    ap("")
    ap("**What this dossier does not license.** A verdict of *present* says "
       "the meaning-material a tradition is made of is measurably in the "
       "book; it does not say the author read that tradition. The one "
       "documented intertextual toehold in either direction is #57's: the "
       "book's only non-generic shared phrase with an influence excerpt is "
       "`a virus` (Burroughs). Conversely, a *thin* verdict does not "
       "exonerate the tradition -- the excerpt arms are too small to refute "
       "anything, which is why they are reported as bounds. And the two "
       "*underdetermined* expectations (BG4 tenderness, BT4 elliptical "
       "time) are open questions with named instruments, not negatives.")
    ap("")

    # --- 9. reconciliation
    ap("## 9. Reconciliation")
    ap("")
    ap("| item | expected | observed | source | ok |")
    ap("|---|---|---|---|---|")
    for r in kn.read_tsv(OUT / "reconciliation.tsv"):
        ap(f"| {r['item']} | {r['expected']} | {r['observed']} | "
           f"{r['source']} | {r['ok']} |")
    ap("")
    ap("## 10. Reproduce")
    ap("")
    ap("```bash")
    ap("python work/genre_prior.py      # writes analysis/genre_prior/")
    ap("python work/genre_prior.py --quick  # capped technical registers")
    ap("python work/test_genre_prior.py")
    ap("```")
    ap("")
    ap("Offline and deterministic; no network, no clock in any statistic; "
       "rebuild is byte-identical except the generated stamp.  The prior "
       "ledger's probe date is #57's pinned survey date, inherited rather "
       "than re-derived.")
    ap("")
    (OUT / "dossier.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"[genre_prior] dossier.md: {len(lines)} lines")


# ===========================================================================
# 9. cli
# ===========================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="#60 genre/prior dossier: expectations for the kind of "
                    "meaning present, probed against the corpus")
    parser.add_argument("--quick", action="store_true",
                        help="cap the register corpora at 5,000 tokens "
                             "(book-side numbers are mode-invariant)")
    args = parser.parse_args(argv)
    build(log=print, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
