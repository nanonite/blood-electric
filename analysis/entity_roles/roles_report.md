# Entity roles from the jacket synopsis, verified against the text (#59)

Generated: 2026-10-14 16:55 UTC

Nine roles are read off the back-cover synopsis (p224) and tested against the book text, pages 1-223 -- the synopsis page is excluded from its own check. Every verdict is computed from counts; every evidenced role carries the citations that earned it. **6 evidenced / 2 uncertain / 1 refuted.**

## 1. Method

* **Seed.** Each role cites the synopsis sentence(s) it is read from (shared with #66's `SYNOPSIS_CLAIMS`, so the text cannot drift). No other source seeds a role.
* **Evidence.** Presence, recurrence, centrality, the #30 role category, clause co-occurrence and actor/verb events are recomputed over the frozen #17/#33/#30/#27 layers with #66's primitives. A probe is *defining* or *supporting*.
* **Verdict.** A role is `evidenced` when every defining probe passes, `uncertain` when some but not all do, and `refuted` when no *substantive* defining probe passes. Presence is a precondition, not a role: a role that exists but whose specific relation is only partly attested is `uncertain` (the entity is real, the predicate is not), while one whose every asserted property fails is `refuted`.

## 2. The role table

| id | entity | synopsis role | claims | #30 role | verdict | mentions | pages | events |
|---|---|---|---|---|---|---:|---:|---:|
| R1 | ADAM / ADAM Doll | central being (creature) whose circuitry the collusion ignites | S1/S4 | creature/persona | **evidenced** | 165 | 81 | 603 |
| R2 | ANTi-ADAM | antagonist clashing with ADAM | S4 | antagonist | **uncertain** | 6 | 6 | 3 |
| R3 | Cadaver City | setting / place of the fatal collusion | S1 | place | **evidenced** | 191 | 118 | 352 |
| R4 | Placenta World | setting / place of the dogs' gene war | S2 | place | **evidenced** | 57 | 44 | 103 |
| R5 | Sato Corporation | organization behind the digital vampires, napalm torture and its victims | S3/S5 | organization | **evidenced** | 24 | 23 | 69 |
| R6 | the dog (dogs of zero) | war-making force waging gene war | S2 | referent | **uncertain** | 459 | 188 | 877 |
| R7 | Super Cherry 666 | robosuccubus hunter of the corporation's victims | S5 | concept | **refuted** | 3 | 3 | 6 |
| R8 | EVOL | pink-ash planet (setting) | S4 | persona | **evidenced** | 13 | 12 | 16 |
| R9 | digital vampires | agents of Sato Corporation | S3 | persona | **evidenced** | 13 | 13 | 21 |

## 3. Evidence, per role

### R1 ADAM / ADAM Doll -- **evidenced** (4/4 defining probes)

Synopsis role: *central being (creature) whose circuitry the collusion ignites* (claims S1|S4, #30/#66 role `creature/persona`). 165 mentions on 81 pages, 603 actor/target events.

* **PASS** `presence of the ADAM family` -- mentions of {ADAM|ADAM Doll}, observed 165 (wanted >=1).
* **PASS** `recurrence across the book` -- pages carrying {ADAM|ADAM Doll}, observed 81 (wanted >=20).
* **PASS** `centrality: >= 100 actor/target events` -- actor/target events for {ADAM|ADAM Doll}, observed 603 (wanted >=100). [p15:924; p15:947; p16:238]
* **PASS** `category in {creature, persona}` -- #30/#66 role of {ADAM|ADAM Doll}, observed creature/persona (wanted in {creature|persona}).
* **PASS** `ignition/burn event for the family` -- {ADAM|ADAM Doll} as actor/target of {burn|burning|burns|burnt|ignite|ignited|ignites}, observed 3 (wanted >=1). [p74:405; p75:569; p217:1936]
* **PASS** `ADAM Doll co-occurs with Cadaver City` -- clauses with both ADAM Doll and Cadaver City, observed 4 (wanted >=1). [p72:1611; p116:1344; p142:475]

Citations: p15:924; p15:947; p16:238; p74:405

### R2 ANTi-ADAM -- **uncertain** (2/3 defining probes)

Synopsis role: *antagonist clashing with ADAM* (claims S4, #30/#66 role `antagonist`). 6 mentions on 6 pages, 3 actor/target events.

* **PASS** `presence of ANTi-ADAM` -- mentions of {ANTI-ADAM}, observed 6 (wanted >=1).
* **PASS** `category in {antagonist, persona}` -- #30/#66 role of {ANTI-ADAM}, observed antagonist (wanted in {antagonist|persona}).
* **FAIL** `clash/hostile event between ADAM and ANTi-ADAM` -- event between {ADAM|ADAM Doll} and {ANTI-ADAM}, observed 0 (wanted >=1).
* **PASS** `ADAM / ANTi-ADAM co-occur in a clause` -- clauses with both ADAM and ANTI-ADAM, observed 1 (wanted >=1). [p216:1132]
* **PASS** `hostile actor/target events` -- hostile actor/target events for {ANTI-ADAM}, observed 1 (wanted >=1). [p144:1521]

Citations: p216:1132; p144:1521

### R3 Cadaver City -- **evidenced** (3/3 defining probes)

Synopsis role: *setting / place of the fatal collusion* (claims S1, #30/#66 role `place`). 191 mentions on 118 pages, 352 actor/target events.

* **PASS** `presence of Cadaver City` -- mentions of {Cadaver City}, observed 191 (wanted >=1).
* **PASS** `recurrence >= 3 pages` -- pages carrying {Cadaver City}, observed 118 (wanted >=3).
* **PASS** `category place` -- #30/#66 role of {Cadaver City}, observed place (wanted in {place}).
* **PASS** `ADAM Doll co-occurs with Cadaver City` -- clauses with both ADAM Doll and Cadaver City, observed 4 (wanted >=1). [p72:1611; p116:1344; p142:475]
* **PASS** `embryo family co-occurs with Cadaver City` -- clauses with both embryo and Cadaver City, observed 18 (wanted >=1). [p33:1240; p67:1883; p72:1611]
* **PASS** `angel family co-occurs with Cadaver City` -- clauses with both angel and Cadaver City, observed 9 (wanted >=1). [p67:1883; p72:1611; p77:961]

Citations: p72:1611; p116:1344; p142:475; p33:1240

### R4 Placenta World -- **evidenced** (3/3 defining probes)

Synopsis role: *setting / place of the dogs' gene war* (claims S2, #30/#66 role `place`). 57 mentions on 44 pages, 103 actor/target events.

* **PASS** `presence of Placenta World` -- mentions of {Placenta World}, observed 57 (wanted >=1).
* **PASS** `recurrence >= 3 pages` -- pages carrying {Placenta World}, observed 44 (wanted >=3).
* **PASS** `category place` -- #30/#66 role of {Placenta World}, observed place (wanted in {place}).
* **PASS** `the dog co-occurs with Placenta World` -- clauses with both the dog and Placenta World, observed 2 (wanted >=1). [p62:1105; p194:881]
* **FAIL** `the dog wages war in Placenta World` -- {the dog} as actor/target of {wage|waged|wages|waging}, observed 0 (wanted >=1).

Citations: p62:1105; p194:881

### R5 Sato Corporation -- **evidenced** (5/5 defining probes)

Synopsis role: *organization behind the digital vampires, napalm torture and its victims* (claims S3|S5, #30/#66 role `organization`). 24 mentions on 23 pages, 69 actor/target events.

* **PASS** `presence of Sato Corporation` -- mentions of {Sato Corporation|Sato}, observed 24 (wanted >=1).
* **PASS** `category organization` -- #30/#66 role of {Sato Corporation|Sato}, observed organization (wanted in {organization}).
* **PASS** `napalm family co-occurs with torture family` -- clauses with both napalm and torture, observed 6 (wanted >=1). [p83:415; p89:1259; p119:60]
* **PASS** `torture family co-occurs with victim family` -- clauses with both torture and victim, observed 7 (wanted >=1). [p83:415; p119:60; p130:2]
* **PASS** `Sato Corporation co-occurs with victim family` -- clauses with both Sato Corporation and victim, observed 6 (wanted >=1). [p79:1396; p83:415; p147:2]
* **PASS** `vampire family co-occurs with the corporation` -- clauses with both vamp and Sato Corporation, observed 5 (wanted >=1). [p39:1114; p87:537; p161:396]
* **PASS** `chaos family co-occurs with vampire family` -- clauses with both chaos and vamp, observed 3 (wanted >=1). [p71:1521; p82:1177; p177:685]
* **PASS** `hostile actor/target events` -- hostile actor/target events for {Sato Corporation|Sato}, observed 7 (wanted >=1). [p25:254; p41:199; p53:1687]

Citations: p83:415; p89:1259; p119:60; p130:2

### R6 the dog (dogs of zero) -- **uncertain** (2/3 defining probes)

Synopsis role: *war-making force waging gene war* (claims S2, #30/#66 role `referent`). 459 mentions on 188 pages, 877 actor/target events.

* **PASS** `presence of the dog referent` -- mentions of {the dog}, observed 459 (wanted >=1).
* **PASS** `recurrence >= 20 pages` -- pages carrying {the dog}, observed 188 (wanted >=20).
* **FAIL** `the dog wages war (wage-family verb)` -- {the dog} as actor/target of {wage|waged|wages|waging}, observed 0 (wanted >=1).
* **PASS** `the dog co-occurs with the zero family` -- clauses with both the dog and zero, observed 32 (wanted >=1). [p8:485; p15:689; p21:1510]
* **PASS** `hostile actor/target events` -- hostile actor/target events for {the dog}, observed 72 (wanted >=1). [p7:324; p10:1007; p12:268]

Citations: p8:485; p15:689; p21:1510; p7:324

### R7 Super Cherry 666 -- **refuted** (1/4 defining probes)

Synopsis role: *robosuccubus hunter of the corporation's victims* (claims S5, #30/#66 role `concept`). 3 mentions on 3 pages, 6 actor/target events.

* **PASS** `presence of Super Cherry 666` -- mentions of {Super Cherry 666}, observed 3 (wanted >=1).
* **FAIL** `category in {creature, persona}` -- #30/#66 role of {Super Cherry 666}, observed concept (wanted in {creature|persona}).
* **FAIL** `Super Cherry 666 co-occurs with the succubus family` -- clauses with both Super Cherry 666 and succubus, observed 0 (wanted >=1).
* **FAIL** `Super Cherry 666 takes a hunt verb` -- {Super Cherry 666} as actor/target of {hunt|hunted|hunting|hunts}, observed 0 (wanted >=1).
* **FAIL** `Super Cherry 666 co-occurs with Sato Corporation` -- clauses with both Super Cherry 666 and Sato Corporation, observed 0 (wanted >=1).

### R8 EVOL -- **evidenced** (3/3 defining probes)

Synopsis role: *pink-ash planet (setting)* (claims S4, #30/#66 role `persona`). 13 mentions on 12 pages, 16 actor/target events.

* **PASS** `presence of EVOL` -- mentions of {EVOL}, observed 13 (wanted >=1).
* **PASS** `recurrence >= 3 pages` -- pages carrying {EVOL}, observed 12 (wanted >=3).
* **PASS** `EVOL co-occurs with the planet family` -- clauses with both EVOL and planet, observed 6 (wanted >=1). [p77:111; p79:1396; p111:475]
* **PASS** `EVOL co-occurs with the pink family` -- clauses with both EVOL and pink, observed 5 (wanted >=1). [p79:1396; p132:767; p153:2045]
* **PASS** `EVOL co-occurs with the ash family` -- clauses with both EVOL and ash, observed 4 (wanted >=1). [p79:1396; p153:2045; p173:904]

Citations: p77:111; p79:1396; p111:475; p132:767

### R9 digital vampires -- **evidenced** (3/3 defining probes)

Synopsis role: *agents of Sato Corporation* (claims S3, #30/#66 role `persona`). 13 mentions on 13 pages, 21 actor/target events.

* **PASS** `presence of the vamp family entity` -- mentions of {vamp}, observed 13 (wanted >=1).
* **PASS** `recurrence >= 3 pages` -- pages carrying {vamp}, observed 13 (wanted >=3).
* **PASS** `vampire family co-occurs with the corporation` -- clauses with both vamp and Sato Corporation, observed 5 (wanted >=1). [p39:1114; p87:537; p161:396]
* **PASS** `category in {persona, concept, creature}` -- #30/#66 role of {vamp}, observed persona (wanted in {concept|creature|persona}).
* **PASS** `hostile actor/target events` -- hostile actor/target events for {vamp}, observed 2 (wanted >=1). [p54:1312; p116:1067]

Citations: p39:1114; p87:537; p161:396; p54:1312

## 4. Refutations and open questions

* **R2 ANTi-ADAM -- uncertain.** clash/hostile event between ADAM and ANTi-ADAM [pair_verb] (0 observed, wanted >=1)
* **R6 the dog (dogs of zero) -- uncertain.** the dog wages war (wage-family verb) [any_verb] (0 observed, wanted >=1)
* **R7 Super Cherry 666 -- refuted.** category in {creature, persona} [category] (concept observed, wanted in {creature|persona}); Super Cherry 666 co-occurs with the succubus family [cooccur] (0 observed, wanted >=1); Super Cherry 666 takes a hunt verb [any_verb] (0 observed, wanted >=1)

The Super Cherry 666 case is the sharp one: the hunting formula the jacket names is real, but its actor is ToKAGE, and Super Cherry 666 never hunts. ANTi-ADAM exists and is the only entity #30 calls an `antagonist`, but no clash event links it to ADAM; the dog is a recurrent, violent actor, but no `wage`-family verb stands with it. The synopsis names the right entities and the wrong predicates.

## 5. Reconciliation

| quantity | owner | here | match |
|---|---|---|---|
| #30 accepted entities in the glossary (rows read) | 98 | 98 | ok |
| #66 mention count for ADAM | 129 | 129 | ok |
| #66 mention count for ADAM Doll | 165 | 165 | ok |
| #66 mention count for ANTI-ADAM | 6 | 6 | ok |
| #66 mention count for Cadaver City | 191 | 191 | ok |
| #66 mention count for Placenta World | 57 | 57 | ok |
| #66 mention count for Sato Corporation | 24 | 24 | ok |
| #66 mention count for Sato | 3 | 3 | ok |
| #66 mention count for the dog | 459 | 459 | ok |
| #66 mention count for Super Cherry 666 | 3 | 3 | ok |
| #66 mention count for EVOL | 13 | 13 | ok |
| #66 mention count for vamp | 13 | 13 | ok |
| #66 event rows re-derived | 4829 | 4829 | ok |
| p224 synopsis page excluded from the check | 224 | excluded (1 clauses) | ok |
| screened units = assigned + unassigned | 73840 | 73840 | ok |

## 6. Reproduce

```bash
python work/entity_roles.py      # writes analysis/entity_roles/
python work/test_entity_roles.py
```

Offline and deterministic; rebuild is byte-identical except the generated stamp.
