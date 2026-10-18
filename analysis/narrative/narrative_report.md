# Narrative reconstruction (#66) -- entity/event tracking over the AST

Generated: 2026-10-14 16:29 UTC

The narrative skeleton of *Blood Electric*, recovered from the text: who does what to whom, event by event, every row citation-linked to a #33 clause (and through its clause_id to the #62 interlinear edition). The jacket synopsis is used only as a sanity check on the skeleton, never as a script.

## 1. Inputs (all frozen artifacts)

| layer | artifact | owner |
|---|---|---|
| unit stream | analysis/tokenizer/units.tsv | #17 |
| clause / AST utterance partition | analysis/clauses/clauses.tsv | #33 (built on the #14 AST) |
| dramatis personae + roles + surfaces | analysis/entity/entity_glossary.tsv | #30 |
| frozen verb lexicon | analysis/collocation/verb_lexicon.tsv | #27 |
| interlinear cross-link | analysis/interlinear/interlinear.tsv | #62 |

## 2. Method

* **Mention layer.** Every #30 glossary surface (single units and adjacent-unit phrase runs, longest match first, so 'ADAM Doll' wins over 'ADAM') plus the two disclosed referents the book personifies but #30 correctly refuses as ordinary English words: 'the dog' ('dog'/'dogs', exact-unit match) and 'self'. Glued compounds ('dog-strand', 'self-ruin') stay their own surfaces and are not counted as referent mentions.
* **Verb layer.** The frozen #27 lexicon. brown/book/morph sources are high confidence; the bare -s/-es signal (morph_s, ambiguous with plural nouns) is low and can never reach high.
* **Event rule.** Per clause, for each verb: the nearest mention to the LEFT is the actor, the nearest to the RIGHT is the target (#27/#30's does/done-to asymmetry), within 12 units of the actor and 8 of the target. A mention spanning the verb stands on neither side. Patterns: AVT (both sides), AV, VT.
* **Confidence.** high-confidence verb + max gap <= 4 = high; <= 8 = medium; beyond = low. A morph_s verb caps at medium (gap <= 4). Copulas (is/was/...) are flagged in verb_class so the action skeleton can be read without them.
* **Sanity check.** Each jacket-synopsis claim (p224) is re-tested against the book text, pages 1-223 only -- the synopsis page is excluded from its own check -- and verdicted supported / partial / unsupported from disclosed element counts.

## 3. Headline numbers

| quantity | value |
|---|---:|
| clauses | 5,613 |
| screened_units | 73,840 |
| units_assigned_to_clauses | 73,840 |
| units_unassigned | 0 |
| mentions | 2,792 |
| events | 4,829 |
| events AVT / AV / VT | 501 / 2,570 / 1,758 |
| events high / medium / low | 2,131 / 1,954 / 744 |
| action / copula events | 4,158 / 671 |
| actors with events / targets with events | 81 / 79 |
| referent mentions (the dog / self) | 459 / 364 |

## 4. The dramatis personae (mention and event counts)

Roles are #30's; the event columns are this build's. Two counting notes: a phrase mention consumes its constituent units, so 'Sato Corporation' does not also count as a 'Sato' mention and 'ADAM Doll' does not count as 'ADAM' (#30's unit-level occurrences stay in the reconciliation table); and the back-cover synopsis page (p224) is excluded from these layers, so the profiles span the book text, pages 1-223.

| entity | kind | role | mentions | pages | first-last | events | does | done-to |
|---|---|---|---:|---|---:|---|---:|---:|---:|
| the dog | referent | referent | 459 | 188 | p4-p223 | 877 | 513 | 372 |
| self | referent | referent | 364 | 121 | p4-p223 | 696 | 439 | 268 |
| cadaver | glossary | persona | 237 | 107 | p5-p223 | 396 | 234 | 162 |
| Cadaver City | glossary | place | 191 | 118 | p5-p223 | 352 | 190 | 162 |
| ADAM Doll | glossary | creature | 165 | 61 | p43-p219 | 342 | 217 | 127 |
| retro-ADAM | glossary | persona | 165 | 43 | p87-p176 | 315 | 170 | 145 |
| ADAM | glossary | persona | 129 | 81 | p15-p216 | 262 | 155 | 107 |
| ToKAGE | glossary | persona | 127 | 84 | p4-p222 | 244 | 133 | 112 |
| DNA | glossary | persona | 59 | 48 | p11-p219 | 153 | 96 | 57 |
| DNA=channel | glossary | machine-system | 75 | 55 | p6-p222 | 129 | 75 | 54 |
| android | glossary | persona | 63 | 48 | p13-p219 | 128 | 82 | 46 |
| hydromania | glossary | persona | 60 | 52 | p11-p208 | 115 | 69 | 46 |
| Placenta World | glossary | place | 57 | 44 | p5-p222 | 103 | 47 | 56 |
| BABEL | glossary | persona | 47 | 42 | p9-p218 | 89 | 46 | 43 |
| HDD | glossary | persona | 35 | 30 | p17-p208 | 66 | 40 | 26 |
| VTR | glossary | persona | 32 | 28 | p7-p223 | 66 | 36 | 30 |
| XX | glossary | persona | 26 | 25 | p4-p222 | 66 | 45 | 22 |
| Sato Corporation | glossary | organization | 24 | 23 | p5-p221 | 63 | 31 | 32 |
| SADO | glossary | persona | 29 | 26 | p14-p217 | 58 | 26 | 32 |
| NDROID | glossary | persona | 20 | 13 | p110-p142 | 49 | 33 | 16 |
| SODO | glossary | persona | 22 | 22 | p40-p215 | 45 | 27 | 18 |
| deathshead | glossary | persona | 26 | 23 | p44-p218 | 41 | 26 | 15 |
| SM | glossary | persona | 23 | 21 | p6-p222 | 40 | 21 | 19 |
| BABEL=TV | glossary | machine-system | 16 | 14 | p22-p205 | 36 | 21 | 15 |
| NDRO | glossary | persona | 23 | 20 | p4-p221 | 34 | 15 | 19 |
| okama | glossary | persona | 18 | 17 | p7-p223 | 31 | 19 | 12 |
| BLOOD ELECTRIC | glossary | concept | 14 | 14 | p1-p223 | 30 | 20 | 10 |
| Placenta | glossary | persona | 13 | 12 | p53-p216 | 28 | 18 | 10 |
| dustNirverna | glossary | persona | 15 | 12 | p36-p193 | 26 | 13 | 13 |
| Cadaver City-city | glossary | concept | 12 | 12 | p4-p220 | 25 | 11 | 14 |
| cyberBuddha | glossary | persona | 14 | 11 | p35-p219 | 25 | 18 | 7 |
| HIV | glossary | persona | 13 | 9 | p167-p175 | 23 | 14 | 9 |
| TV | glossary | persona | 7 | 7 | p15-p218 | 22 | 17 | 5 |
| vamp | glossary | persona | 13 | 13 | p7-p223 | 21 | 8 | 13 |
| NIHIL | glossary | persona | 9 | 8 | p21-p138 | 20 | 13 | 7 |
| Sato | glossary | organization | 3 | 3 | p53-p95 | 6 | 5 | 1 |
| Super Cherry 666 | glossary | concept | 3 | 3 | p56-p79 | 6 | 4 | 2 |
| ANTI-ADAM | glossary | antagonist | 6 | 6 | p128-p219 | 3 | 1 | 2 |

## 5. Who does what to whom (the skeleton)

Per-actor profile over the event table; verbs are the top action and copula partners mixed, with the example citations showing the actor -> verb -> target triple behind them.

* **ADAM** (glossary / persona) -- 129 mentions on 81 pages (p15-p216), 262 events
  * does: is(14),  was(9),  fecundates(6),  defleshed(4),  masses(4)
  * done-to: murder(7),  crucified(5),  line(5),  black(4),  invades(3)
  * acts on: the dog(8),  self(4),  TV(3),  ToKAGE(3),  BABEL(2)
  * acted on by: self(7),  the dog(4),  BABEL(2),  Cadaver City(2),  DNA=channel(1)
  * evidence: p15:924 ADAM->masses->-,  p15:947 ADAM->game->-,  p16:238 -->rapes->ADAM

* **ANTI-ADAM** (glossary / antagonist) -- 6 mentions on 6 pages (p128-p219), 3 events
  * does: murder(1)
  * done-to: disappears(1),  receiving(1)
  * acted on by: ADAM Doll(1)
  * evidence: p128:956 -->receiving->ANTI-ADAM,  p144:1521 ANTI-ADAM->murder->-,  p219:1356 ADAM Doll->disappears->ANTI-ADAM

* **Super Cherry 666** (glossary / concept) -- 3 mentions on 3 pages (p56-p79), 6 events
  * does: were(2),  resolved(1),  stimulated(1)
  * done-to: bodies(1),  like(1)
  * evidence: p56:904 -->like->Super Cherry 666,  p56:919 -->bodies->Super Cherry 666,  p56:969 Super Cherry 666->were->-

* **Sato Corporation** (glossary / organization) -- 24 mentions on 23 pages (p5-p221), 63 events
  * does: victims(5),  hunting(4),  assassins(2),  mass(2),  murder(2)
  * done-to: hunting(5),  skulls(4),  controlled(2),  is(2),  searching(2)
  * acts on: Cadaver City(6),  NEMESIS(2),  ADAM(1),  EVOL(1)
  * acted on by: self(4),  retro-ADAM(3),  ToKAGE(2)
  * evidence: p11:498 -->were->Sato Corporation,  p11:503 -->connected->Sato Corporation,  p25:232 -->animals->Sato Corporation

* **Sato** (glossary / organization) -- 3 mentions on 3 pages (p53-p95), 6 events
  * does: corps(1),  fecundates(1),  kills(1),  murder(1),  rape(1)
  * done-to: dogs(1)
  * evidence: p53:1681 Sato->corps->-,  p53:1687 Sato->rape->-,  p56:1706 Sato->kills->-

* **the dog** (referent / referent) -- 459 mentions on 188 pages (p4-p223), 877 events
  * does: is(40),  was(36),  pack(28),  sun(19),  murder(15)
  * done-to: murder(24),  is(18),  like(15),  sun(11),  black(9)
  * acts on: self(19),  the dog(8),  ADAM Doll(6),  cadaver(6),  ADAM(4)
  * acted on by: self(16),  ADAM Doll(15),  retro-ADAM(11),  cadaver(10),  ADAM(8)
  * evidence: p4:137 -->like->the dog,  p4:155 the dog->was->-,  p4:159 the dog->done->-

* **self** (referent / referent) -- 364 mentions on 121 pages (p4-p223), 696 events
  * does: is(45),  was(33),  fecundates(17),  murder(12),  sun(11)
  * done-to: murder(15),  was(15),  is(11),  game(8),  desire(6)
  * acts on: the dog(16),  self(11),  ADAM(7),  Placenta World(5),  Sato Corporation(4)
  * acted on by: the dog(19),  self(11),  ADAM(4),  cadaver(4),  BABEL=TV(3)
  * evidence: p4:645 self->caused->-,  p4:669 self->rape->-,  p4:674 self->function->-

* **ADAM Doll** (glossary / creature) -- 165 mentions on 61 pages (p43-p219), 342 events
  * does: was(16),  is(12),  sun(10),  black(8),  like(7)
  * done-to: like(8),  murder(8),  is(6),  desert(5),  line(5)
  * acts on: the dog(15),  ToKAGE(7),  Placenta(3),  cadaver(3),  ADAM Doll(2)
  * acted on by: the dog(6),  DNA(5),  ADAM Doll(2),  NDRO(2),  ToKAGE(2)
  * evidence: p43:1157 -->grasps->ADAM Doll,  p43:1204 ADAM Doll->escapes->-,  p44:294 -->records->ADAM Doll

* **ToKAGE** (glossary / persona) -- 127 mentions on 84 pages (p4-p222), 244 events
  * does: is(18),  was(10),  fecundates(7),  black(3),  down(3)
  * done-to: sun(7),  crucified(5),  love(5),  speed(5),  fecundates(4)
  * acts on: BABEL(4),  vamp(3),  ADAM Doll(2),  Cadaver City-city(2),  Placenta(2)
  * acted on by: ADAM Doll(7),  XX(4),  ADAM(3),  the dog(3),  DNA(2)
  * evidence: p4:549 ToKAGE->crowds->Cadaver City-city,  p5:347 ToKAGE->is->-,  p6:735 ToKAGE->polarized->-

* **cadaver** (glossary / persona) -- 237 mentions on 107 pages (p5-p223), 396 events
  * does: was(23),  is(21),  dogs(9),  sun(9),  creatures(5)
  * done-to: fecundates(14),  sun(9),  is(7),  murder(7),  screen(6)
  * acts on: retro-ADAM(13),  the dog(10),  self(4),  SM(3),  hydromania(3)
  * acted on by: retro-ADAM(12),  XX(7),  the dog(6),  self(4),  ADAM Doll(3)
  * evidence: p5:2366 -->scream->cadaver,  p5:2395 cadaver->is->-,  p6:1808 self->murder->cadaver

* **DNA=channel** (glossary / machine-system) -- 75 mentions on 55 pages (p6-p222), 129 events
  * does: murder(6),  was(6),  is(4),  exploded(3),  sun(3)
  * done-to: fecundates(4),  was(4),  sun(3),  abolished(2),  accelerates(2)
  * acts on: self(3),  ADAM(1),  Placenta World(1),  retro-ADAM(1),  the dog(1)
  * acted on by: retro-ADAM(4),  ADAM(2),  cadaver(2),  self(2),  the dog(2)
  * evidence: p6:1287 DNA=channel->was->-,  p6:1291 DNA=channel->paralyzed->-,  p6:1304 DNA=channel->murder->-

* **Placenta World** (glossary / place) -- 57 mentions on 44 pages (p5-p222), 103 events
  * does: was(6),  murder(4),  spot(3),  is(2),  line(2)
  * done-to: murder(4),  is(3),  end(2),  fills(2),  mass(2)
  * acts on: retro-ADAM(4),  ADAM(1),  DNA(1),  ToKAGE(1),  hydromania(1)
  * acted on by: Cadaver City(6),  self(5),  ADAM Doll(2),  NDRO(2),  NDROID(2)
  * evidence: p5:759 Placenta World->was->-,  p5:1370 NDRO->rapes->Placenta World,  p6:1449 self->mass->Placenta World

* **Cadaver City-city** (glossary / concept) -- 12 mentions on 12 pages (p4-p220), 25 events
  * does: is(2),  broken(1),  desert(1),  down(1),  drew(1)
  * done-to: crowds(2),  animals(1),  awakes(1),  control(1),  excoriate(1)
  * acted on by: the dog(3),  ToKAGE(2)
  * evidence: p4:549 ToKAGE->crowds->Cadaver City-city,  p24:1983 -->animals->Cadaver City-city,  p24:1993 -->streams->Cadaver City-city

### The narrative backbone: strongest directed pairs

| actor | target | n_events | top verbs | first-last |
|---|---|---:|---|---|
| the dog | self | 19 | caused(2),  fecundates(2),  line(2),  streams(2),  control(1) | p4:1294 - p222:1000 |
| self | the dog | 16 | digested(2),  fires(2),  like(2),  abolished(1),  animals(1) | p4:1006 - p222:910 |
| ADAM Doll | the dog | 15 | black(2),  commits(1),  desire(1),  drinks(1),  escape(1) | p49:1076 - p217:796 |
| cadaver | retro-ADAM | 13 | murder(2),  was(2),  accelerates(1),  break(1),  down(1) | p144:284 - p174:1351 |
| retro-ADAM | cadaver | 12 | accelerates(1),  decayed(1),  dogs(1),  engages(1),  game(1) | p145:2250 - p171:1722 |
| retro-ADAM | the dog | 11 | circuits(1),  condition(1),  fecundates(1),  infected(1),  infested(1) | p124:416 - p174:2414 |
| self | self | 11 | was(3),  abolished(1),  black(1),  cable(1),  despair(1) | p34:2155 - p124:1082 |
| cadaver | the dog | 10 | scanned(2),  circulates(1),  desire(1),  form(1),  mechanisms(1) | p7:2351 - p223:2351 |
| ADAM | the dog | 8 | is(2),  black(1),  jointed(1),  masses(1),  occupies(1) | p16:1291 - p203:337 |
| the dog | the dog | 8 | controlled(2),  like(2),  was(2),  level(1),  screamed(1) | p24:886 - p100:627 |
| ADAM Doll | ToKAGE | 7 | evolves(1),  exploded(1),  forgets(1),  leering(1),  like(1) | p57:2026 - p99:1086 |
| XX | cadaver | 7 | is(2),  like(2),  eroded(1),  fecundates(1),  sun(1) | p42:667 - p218:952 |
| self | ADAM | 7 | controls(1),  game(1),  invades(1),  line(1),  murder(1) | p132:107 - p208:65 |
| Cadaver City | Placenta World | 6 | fire(1),  form(1),  gang(1),  gangs(1),  murder(1) | p200:1442 - p214:592 |
| Sato Corporation | Cadaver City | 6 | assassins(2),  hunting(2),  murder(2) | p41:161 - p113:957 |
| the dog | ADAM Doll | 6 | analyzes(1),  back(1),  broke(1),  desert(1),  rape(1) | p57:1023 - p178:1096 |
| the dog | cadaver | 6 | emulated(1),  installed(1),  mass(1),  pack(1),  registers(1) | p120:1638 - p176:842 |
| DNA | ADAM Doll | 5 | channel(1),  controlled(1),  like(1),  speed(1),  was(1) | p72:975 - p142:1347 |
| hydromania | retro-ADAM | 5 | accelerates(1),  controls(1),  form(1),  murder(1),  scream(1) | p145:161 - p173:2091 |
| retro-ADAM | Cadaver City | 5 | covered(1),  installed(1),  is(1),  sutures(1),  was(1) | p126:597 - p173:2444 |
| self | Placenta World | 5 | mass(2),  dogs(1),  fills(1),  invades(1) | p6:1449 - p222:1449 |
| ADAM | self | 4 | deranged(1),  functions(1),  links(1),  were(1) | p33:1078 - p134:1243 |
| Placenta World | retro-ADAM | 4 | head(1),  line(1),  murder(1),  structure(1) | p158:430 - p158:475 |
| ToKAGE | BABEL | 4 | is(1),  plunder(1),  restrained(1),  this(1) | p54:682 - p75:968 |
| XX | ToKAGE | 4 | black(1),  fecundates(1),  space(1),  sun(1) | p115:1871 - p118:745 |

The full table is pairs.tsv (205 directed pairs); every event row is in events.tsv with its citation and clause link.

## 6. The dog and 'self' (the referent layer)

Neither is a #30 entity -- both are ordinary English words the book personifies -- so they are tracked as disclosed referents rather than smuggled into the glossary. Exact-unit matching only.

* **the dog** (referent / referent) -- 459 mentions on 188 pages (p4-p223), 877 events
  * does: is(40),  was(36),  pack(28),  sun(19),  murder(15)
  * done-to: murder(24),  is(18),  like(15),  sun(11),  black(9)
  * acts on: self(19),  the dog(8),  ADAM Doll(6),  cadaver(6),  ADAM(4)
  * acted on by: self(16),  ADAM Doll(15),  retro-ADAM(11),  cadaver(10),  ADAM(8)
  * evidence: p4:137 -->like->the dog,  p4:155 the dog->was->-,  p4:159 the dog->done->-

* **self** (referent / referent) -- 364 mentions on 121 pages (p4-p223), 696 events
  * does: is(45),  was(33),  fecundates(17),  murder(12),  sun(11)
  * done-to: murder(15),  was(15),  is(11),  game(8),  desire(6)
  * acts on: the dog(16),  self(11),  ADAM(7),  Placenta World(5),  Sato Corporation(4)
  * acted on by: the dog(19),  self(11),  ADAM(4),  cadaver(4),  BABEL=TV(3)
  * evidence: p4:645 self->caused->-,  p4:669 self->rape->-,  p4:674 self->function->-

## 7. Where the events are (20-page arcs)

Entity columns count surface mentions in each page range; n_mentions counts all tracked entities.

| arc | events | mentions | ADAM | ANTi-ADAM | Sato Corporation | Super Cherry 666 | the dog | self | ToKAGE | ADAM Doll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p1-20 | 423 | 252 | 12 | 0 | 2 | 0 | 56 | 72 | 14 | 0 |
| p21-40 | 502 | 262 | 25 | 0 | 2 | 0 | 39 | 72 | 4 | 0 |
| p41-60 | 485 | 215 | 12 | 0 | 3 | 2 | 54 | 12 | 9 | 27 |
| p61-80 | 488 | 274 | 12 | 0 | 4 | 1 | 29 | 4 | 19 | 70 |
| p81-100 | 332 | 211 | 18 | 0 | 2 | 0 | 39 | 44 | 12 | 1 |
| p101-120 | 394 | 225 | 2 | 0 | 1 | 0 | 32 | 23 | 26 | 16 |
| p121-140 | 389 | 233 | 10 | 1 | 1 | 0 | 39 | 46 | 4 | 6 |
| p141-160 | 490 | 294 | 1 | 1 | 2 | 0 | 50 | 10 | 5 | 9 |
| p161-180 | 510 | 310 | 2 | 0 | 4 | 0 | 58 | 8 | 3 | 8 |
| p181-200 | 337 | 205 | 21 | 0 | 2 | 0 | 29 | 24 | 15 | 4 |
| p201-220 | 378 | 244 | 14 | 4 | 0 | 0 | 24 | 23 | 14 | 24 |
| p221-225 | 101 | 67 | 0 | 0 | 1 | 0 | 10 | 26 | 2 | 0 |

## 8. Jacket-synopsis sanity check (p224)

The back-cover synopsis claims, each re-tested against the book text (pages 1-223; the synopsis page is excluded from its own check). Element-level evidence is in synopsis_elements.tsv.

| claim | synopsis text | elements found | verdict | missing |
|---|---|---|---|---|
| S1 | A fatal collusion of drag embryos and DNA angels in Cadaver City ignites the circuitry of the ADAM Doll | 8/8 | supported | -- |
| S2 | dogs of zero waging gene war in Placenta World | 6/7 | partial | event: {the dog|Placenta World} x {wage|waging|waged} |
| S3 | chaos unleashed by the digital vampires of Sato Corporation | 5/6 | partial | event: {Sato|Sato Corporation} x {unleash|unleashed} |
| S4 | On the pink ash planet EVOL, ADAM and ANTi-ADAM clash in terminal zodiac burn | 3/6 | partial | cooccur: ANTI-ADAM + zodiac; event: {ADAM|ANTI-ADAM} x {clashed|clashing}; event: {ADAM|ANTI-ADAM} x {burn|burns|burning|burnt} |
| S5 | enter robosuccubus Super Cherry 666, hunting for the grotesque nova skulls of Sato Corporation napalm torture victims | 6/9 | partial | cooccur: Super Cherry 666 + succubus; event: {Super Cherry 666} x {hunt|hunts|hunting|hunted}; cooccur: Super Cherry 666 + Sato Corporation |

* **S2 -- partial.** Partly attested: event: {the dog|Placenta World} x {wage|waging|waged}
  * the wage-family verbs never stand with the dog or Placenta World: the synopsis's 'dogs of zero waging gene war' has no matching actor-verb-target event, although the dog+zero and gene+war material does co-occur (32 and 30 clauses).
* **S3 -- partial.** Partly attested: event: {Sato|Sato Corporation} x {unleash|unleashed}
  * no unleash event with the corporation as actor or target; the vamp/Sato Corporation and chaos/vamp material does co-occur (6 and 3 clauses), but the synopsis's verb is absent from the book's event table.
* **S4 -- partial.** Partly attested: cooccur: ANTI-ADAM + zodiac; event: {ADAM|ANTI-ADAM} x {clashed|clashing}; event: {ADAM|ANTI-ADAM} x {burn|burns|burning|burnt}
  * the book's zodiac material sits with ADAM (1 clause), never with ANTi-ADAM, and no clashed/clashing or burn event links the pair; the ADAM+ANTI-ADAM pair itself co-occurs in 1 clause.
* **S5 -- partial.** Partly attested: cooccur: Super Cherry 666 + succubus; event: {Super Cherry 666} x {hunt|hunts|hunting|hunted}; cooccur: Super Cherry 666 + Sato Corporation
  * the hunting formula is attested -- nova+skull, napalm+torture, torture+victim and Sato Corporation+victim all co-occur -- but in the book its actor is ToKAGE (1 'ToKAGE x hunt' event), not Super Cherry 666: the book's 3 mentions of Super Cherry 666 never stand with a hunt verb, a succubus form, or the corporation. The jacket copy attaches the formula to the wrong entrant.

## 9. Limitations

* **Direction is a convention, not a parse.** Left-of-verb = actor, right-of-verb = target inherits #27/#30's does/done-to asymmetry. In the book's genitive chains ('the brain of ADAM murdered X') the rule reads the referent of the chain, not the surface subject; the confidence column prices the distance and the report says so.
* **No coreference.** Mentions are surface matches; a pronoun or a re-naming ('the Doll' after 'ADAM Doll') does not resolve, so the same page can carry an event whose actor is the nearest surface entity rather than the semantic one.
* **The verb lexicon has false friends.** #27's list is morphology-based; 'sun', 'desire' and friends read as verbs here exactly as they did in #30's does/done-to columns, and morph_s hits are priced low. 'clash' itself is not in the lexicon (only 'clashed'/'clashing'), which is why the S4 element says what it says.
* **The synopsis check tests claims, not truth.** An 'unsupported' verdict means the book text does not attest the claim through the disclosed elements; it is a measurement gap, not a refutation of the jacket copy.
* **OCR is the floor**, as everywhere in this workspace: the unit stream is marker OCR of a scan (#4's ensemble bounds the noise).

## 10. Reproduction

    python work/narrative_reconstruction.py                 # full book
    python work/narrative_reconstruction.py --pages 4-6      # a page range
    python work/test_narrative_reconstruction.py

Deterministic: no randomness, no hash-order dependence (all ties broken by a fixed key); the only field that varies between runs is the generated stamp, which no statistic reads.

