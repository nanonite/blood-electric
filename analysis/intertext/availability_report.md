# Intertextual sources: what exists, what is lawful, what was used (#57)

Survey date: **2026-10-14** (pinned; `--probe` re-runs the live requests).  Full machine-readable ledger: `sources.tsv`.

## The five named works are not available as text

The task names *Headcode*, *Human_Worms*, *Smart-d*, *GENEDUB* and *Nonexistence*.  Every one is an in-copyright commercial edition; no rights holder serves the text for reuse.

| work | status | evidence |
|---|---|---|
| Headcode (iUniverse, 2004) | print_or_lending_only | Internet Archive scan headcode0000kenj: access-restricted-item=true, collection printdisabled.  A borrow-only scan, no body fetched.  ISBN 978-0595318254. |
| Human_Worms (2004) | searched_not_found | A publisher/reader spotlight page exists (a critical post and an author interview about the book); it quotes no book text and is not a text source.  No servable edition found. |
| Smart-d (2004) | searched_not_found | No open text or catalogue record with text located. |
| GENEDUB (2006) | searched_not_found | The Great Works author note points to a (now defunct) personal 'Genedub' blog and lists GENEDUB as a publication; no book text is served. |
| Nonexistence (2006) | searched_not_found | No open text or catalogue record with text located. |

**5 named works, 0 ingested.**  This is the negative half of the task and it is a fact about the sources, not a choice about the analysis.  Nothing here substitutes a different text for a named work.

## What was obtained

| source | kind | authorization | how it is used |
|---|---|---|---|
| "abnormal living body of a chemical=anthropoid-modem=heart ..." (Great Works) | this_author_other | open | matcher arm |
| 3AM Magazine: 'THE NUDE BRAIN: AN INTERVIEW WITH KENJI SIRATORI' | author_paratext | open | matcher arm |
| Rhizome 14416 (Wayback capture): 'Interview with Kenji Siratori + Loop Writing' | author_paratext | open | matcher arm |
| Gibson / Stephenson / Sterling (fair-use excerpts) | cited_influence | free_excerpt | matcher arm |
| Burroughs / Artaud / Breton (fair-use excerpts) | cited_influence | free_excerpt | matcher arm |
| Joyce, Finnegans Wake (fair-use excerpts) | cited_influence | free_excerpt | matcher arm |
| General English prose (15 public-domain texts) | register_control | open | matcher arm |
| PubMed abstracts | register_control | rights_verified_per_record | matcher arm |
| Public-domain historical legal/political texts | register_control | open | matcher arm |
| arXiv cs abstracts | register_control | open | matcher arm |

The same-author probe is the *Great Works* text — a **different** Siratori work, freely published online.  It is the only genuine second Siratori text available under the project's research-quotation convention, and it is used only as an idiolect probe.  The cited influences are the #56 fair-use excerpts (Gibson / Stephenson / Sterling; Burroughs / Artaud / Breton; Joyce); they are excerpt-length by copyright necessity, which the matcher discloses with a shuffle null rather than hiding.

## Provenance verified, not asserted

* **The embedded Siratori excerpt is the page it names.**  The first 3 paragraphs of <http://www.greatworks.org.uk/poems/ks.html> re-extract to 330 tokens, sha256 `436ebaddce73d9ca…`.  `--probe` re-fetches the page and diffs the extraction against that pin, so the constant cannot drift silently.
* **Every reference-corpus file is checked against #56's manifest sha256** (30 files); a mismatch is a hard failure in `reconciliation.tsv`, so an arm cannot run on a substituted text.
* **The author arm is the author's own words only.**  #58's paratext table is an interview Q/A; the 1 turn(s) recorded as the interviewer's (PT04) are excluded rather than attributed to the author.

## What was sought and not used

| source | authorization | why not used |
|---|---|---|
| Headcode (iUniverse, 2004) | print_or_lending_only | Internet Archive scan headcode0000kenj: access-restricted-item=true, collection printdisabled.  A borrow-only scan, no body fetched.  ISBN 978-0595318254. |
| Human_Worms (2004) | searched_not_found | A publisher/reader spotlight page exists (a critical post and an author interview about the book); it quotes no book text and is not a text source.  No servable edition found. |
| Smart-d (2004) | searched_not_found | No open text or catalogue record with text located. |
| GENEDUB (2006) | searched_not_found | The Great Works author note points to a (now defunct) personal 'Genedub' blog and lists GENEDUB as a publication; no book text is served. |
| Nonexistence (2006) | searched_not_found | No open text or catalogue record with text located. |
| Rhizome: 'Interview with Kenji Siratori + Loop Writing' (live site) | bot_blocked | The live Rhizome page returns 403 to this agent (recorded by #58); the text is obtained from the Wayback capture above.  Listed so the denial is not silently dropped. |
| Pierre Guyotat (declared influence) | searched_not_found | Named alongside Burroughs in the interviews; no lawful excerpt was assembled, so it is absent from the matcher rather than silently replaced. |
| J. G. Ballard; Aase Berg; Blake Butler; Tetsuo (genre priors) | searched_not_found | Named as genre priors in L7.5.  Priors, not texts; not acquired. |

* Pierre Guyotat, a declared influence in the author interviews, has no lawful excerpt in this workspace.  It is listed as absent rather than silently replaced.
* The Internet Archive `Headcode` scan is lending-only (`access-restricted-item=true`); no body was fetched.

