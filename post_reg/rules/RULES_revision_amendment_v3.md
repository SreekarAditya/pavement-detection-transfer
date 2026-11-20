# Revision analyses: amendment 3 (R5 measures)

Status: approved by the corresponding author on 20 November 2026. This amendment replaces the "Expected answers" and "Reporting groups" sections of amendment 2. It leaves unchanged:
- the clause inventory (tag `post-reg-revision-clauses`);
- the fixtures;
- the frozen evaluator (`qualification-evaluator-v1`);
- the comparison with the Shapes Constraint Language (R6);
- every other section of `RULES_revision.md`.

It is made before any expected answer has been written and before the evaluator has run on any new fixture. It becomes binding when committed and tagged `post-reg-revision-v3`.

## What R5 measures

R5 measures how far the frozen qualification record covers requirements it was not built for, and whether it handles them consistently. Every measure below is decided by the source documents or by a mechanical rule. None depends on a person's reading of a clause.

**M0. Measurable rows.** An inventory row counts as measurable if the cited source text at its locator contains at least one numeric value: a number, a percentage or a time quantity, in digits or words. Rows without one are reported by count and by source, and are excluded from the M1 to M3 denominators. The check is a script committed with this amendment.

**M1. Coverage.** Each measurable clause is encoded in the frozen record and classified as one of:
- expressible unchanged;
- expressible only with a new operator;
- not representable, with the missing capability named.

**Reported:** the count and share in each class, overall and by source.

**M2. Refusal on box-only evidence.** For every measurable clause whose quantity is a physical measurement (length, width, depth, area, percentage of surface, a rating, or a time), the fixture that supplies only image bounding boxes must return "cannot assess".

**Reported:** the share of such fixtures refused, with every exception listed.

**M3. Consistency with the source.** For every encoded clause, the encoded threshold or limit values and the encoded unit must appear in the cited source text at its locator. Number words ("twenty four") and digits are normalized before matching, and unit names and their standard symbols are treated as equivalent. The normalization tables are committed with this amendment.

**Reported:** the share of encodings that pass, with every failure listed and its text shown.

**M4. Unit mismatch.** Every fixture that supplies its measurement in a unit the clause does not use, with no conversion declared, must return "unit mismatch".

**Reported:** the share returned, with every exception listed.

**M5. Agreement with published outcomes.** These cases are compared exactly with the outcomes their sources state:
- the worked-example fixtures (Federal Highway Administration FHWA-HIF-18-022, Tables 1 and 2: 3 clauses, 23 fixtures);
- the registered vendor-guideline verdicts (7).

**Reported:** agreement as n of N for each source. The vendor verdicts are labeled as development cases.

**Boundary fixtures.** These are the fixtures just inside and just outside a threshold. They are executed, and their outputs are published with the encodings, so that any reader can inspect them. They are not used as an accuracy measure: their expected outcome would come from the same encoding that is under test.

## Consequence for the text

- The abstract and the R5 results report M1 (share expressible unchanged, with its count), M2, M3 and M5.
- The registered 12-case transfer result moves to the body as a development result.
- The paper's claim is limited to the coverage, refusal, source consistency and published-outcome agreement measured here.
- The deviations list in the supplement records that this amendment replaced the answer procedure of amendment 2 before any expected answer was written.
