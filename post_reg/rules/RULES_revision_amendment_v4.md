# Revision analyses: amendment 4 (extended comparison types, held-out test)

Status: approved by the corresponding author on 20 November 2026. This amendment adds an extension of the qualification record and a held-out test of it. It changes no earlier result:
- The frozen-record measures of amendment 3 (`qualification-evaluator-v1`; 125 of 564 clauses expressible unchanged) stay reported as measured.
- Every other section of `RULES_revision.md` is unchanged.

The amendment becomes binding when committed and tagged `post-reg-revision-v4`, before any code of the extension is written.

## Purpose

The amendment 3 measures showed two things:
- the record's structure (quantity, unit, threshold, required evidence, action) held 552 of 564 clauses;
- its two comparison types (a not-to-exceed limit and a target band) executed 125.

The remaining clauses use standard requirement forms that the record lacks. This amendment adds those forms as general comparison types, reports them on the 564 clauses **as a development result**, freezes the extended record, and tests it on clauses from sources that were not used to design it.

## E1. The extension (development)

Comparison types are added to the record and evaluator as general constructs. None is written for a particular clause. The cross-record check was motivated by the Route AAA worked examples, so those examples count only as development cases.

| Comparison type | Definition |
| --- | --- |
| Scalar comparison | A value against a threshold with `<`, `<=`, `>`, `>=`, `=` or `!=`. Covers minimum requirements and strict limits. |
| Time limit | The time elapsed between a recorded event and a recorded action, against a limit. Hours, days and weeks convert exactly. Calendar dates and month-day deadlines are compared as dates. A month is treated as a calendar month, never converted to days. |
| Class membership | A recorded code within, or outside, a stated set. |
| Compound condition | All of, any of, or at least k of a list of conditions, each of which is one of the types in this table. |
| Range exclusion | A value outside a stated range, for data-validity checks. |
| Cross-record check | A condition on two or more linked records: the length of a section, or whether the boundaries of two data sets coincide. |

**Direction check (M3b), added to the source-consistency measure.** M3 matches values and units only, so it cannot tell "at least 33%" from "at most 33%". M3b compares each encoded comparison's direction and strictness with the wording of the cited clause, using a committed lexicon:
- "at least", "not less than", "minimum" and "or more" mean `>=`;
- "more than" and "exceeding" mean `>`;
- "not exceeding", "not more than", "maximum", "up to" and "within" mean `<=`;
- "less than" and "below" mean `<`.

An encoding that disagrees is a direction failure. Wording the lexicon does not cover is reported as "direction not determinable". M3b applies to every encoding, in development and in the held-out test.

Encodings of the 564 clauses may be corrected during development, where review of the 114 source-consistency failures (M3) shows an encoding error. Each correction is listed with its reason, and the corrected count is reported beside the original.

**Reported on the 564 clauses** (labeled as a development result):
- M1 to M5 of amendment 3 for the extended record, beside the frozen-record values;
- M5 now includes the 23 Route AAA worked examples, compared exactly with their published outcomes.

## E2. Freeze

When the extension and its tests pass, these are tagged together as `qualification-evaluator-v2`:
- the extended record, schema, operators and evaluator;
- the fixture builder;
- the M0, M3 and M3b scripts;
- the normalization tables and the direction lexicon;
- the duplicate script. Nothing in them changes after this tag. Any later change is a new version, and every fixture is rerun.

## E3. Held-out sources, chosen by rule after the freeze

The held-out set is selected after the `qualification-evaluator-v2` tag, as wide as the selection rules allow, because execution is not the constraint. It has two parts:
1. **Every remaining state program.** This means every one of the 50 states after Connecticut in alphabetical order, Delaware to Wyoming (the District of Columbia and the territories are not included), whose publicly posted program states a numeric acceptance tolerance for automated cracking data, under the approved state selection rule. Every state is checked and logged: included, or skipped with its reason.
2. **Up to 20 further National Highways Authority of India maintenance schedules (Schedule K or E)**, in the order returned by the approved search, not previously used. The cap of 20 is a scope choice, fixed here before any schedule is read.

A candidate that cannot be retrieved from a public copy is skipped, with the reason logged. If a candidate's file was downloaded during earlier intake, that is recorded; its clauses had not been inventoried or encoded.

**Duplicate clauses.** Agency documents reuse template text. Clauses whose normalized text is identical are grouped. Normalization removes whitespace, list and item numbering (for example "(iii)" or "6.4.1"), project names and chainages, and case, and maps units to their standard symbols. Numeric thresholds are kept: "24 hours" and "48 hours" are different clauses. This is done both within the held-out set and against the 564 development clauses, with a script committed with this amendment. A held-out clause identical to a development clause is reported but excluded from the held-out test. **The held-out result is reported on unique clauses**, with the raw count beside it.

**Programs built on a federal template.** Several state programs adapt the federal sample plan (FHWA-HIF-18-032, a development source), so their clauses can paraphrase development clauses without matching them word for word. A held-out program whose own text cites FHWA-HIF-18-032 or calls itself an adaptation of the federal sample plan is flagged by a committed text search. The held-out result is reported with and without the flagged programs.

Each held-out source is inventoried by the approved clause rule. The inventory is committed and tagged `post-reg-revision-heldout-clauses` before any held-out clause is encoded. The fixtures are then built by the same procedure as for the 564 clauses.

## E4. Held-out test

The frozen extended record (`qualification-evaluator-v2`) is run on the held-out clauses.

**Reported:**
- the coverage classes under the frozen extension: executes, needs a comparison type the extension lacks (named), and not representable;
- M0, M2, M3, M3b and M4: measurability, refusal on box-only evidence, source consistency (values and units, then direction) and unit mismatch, each with its count and every exception listed;
- each measure for every held-out source, and pooled over unique clauses;
- the number of sources checked, included and skipped;
- M5 for any published outcome the held-out sources state.

There is no pass threshold. **The held-out values are the test result. The development values are not.**

## Consequence for the text

The paper reports the clause results in this order:
1. **Frozen original record:** 125 of 564 expressible unchanged; structure held 552 of 564. Direction consistency (M3b) is reported for these encodings too.
2. **Extended record on the development clauses**, labeled as development.
3. **Frozen extended record on the held-out clauses**, as the test of generality.

The abstract states the held-out result with its count, beside the original-record result.
