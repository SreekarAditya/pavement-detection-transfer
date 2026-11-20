# Revision analyses: amendment 2 (R5 expected answers)

Status: approved by the corresponding author on 20 November 2026. This amendment changes only Section R5, "Fixtures and expected answers", of `RULES_revision.md` (tag `post-reg-revision-v1`). Every other section of that file, including R1 to R4, R6 and R7 and the runs made under it, is unchanged and stays in force. The amendment becomes binding when it is committed with the files it names and tagged `post-reg-revision-v2`.

## Why the change, and why it is allowed now

The clause inventory, frozen at tag `post-reg-revision-clauses`, holds 564 clauses from the ten listed sources. The fixture file built from it holds 2,574 fixtures, about four or five per clause: inside and outside each threshold, box-only evidence, and mismatched units.

Writing 2,574 expected answers one by one would take about 14 hours. The repetition would invite answering errors that the analysis would then count as evaluator disagreements. The fixtures of one clause follow from a single reading of that clause, so the clause is the natural unit for an answer.

This amendment is made before any expected answer has been written and before the evaluator has run on any new fixture. No result informs it.

## Correction

An earlier version of this amendment recorded that the fixtures predated the clause inventory tag. That was an error: the file times were read in UTC as if they were India Standard Time. In India Standard Time the inventory draft was written at 06:22:14, the inventory was tagged at 06:22:44 and the fixtures were first written at 06:24 on 20 November 2026, after the tag. No deviation occurred.

## Expected answers: one specification per clause, for every clause

1. **Forms.** For every one of the 564 inventoried clauses, an author completes one clause specification form from the source alone, without seeing evaluator output. The form records:
   - whether the inventory row meets the frozen clause rule (`meets_clause_rule`, yes or no), with the author's reason (`clause_rule_reason`);
   - the measured quantity;
   - its unit;
   - the comparison, with whether each boundary is included;
   - the threshold, tolerance or limit;
   - the denominator or aggregation (for example, per 1 km or per section);
   - the evidence the clause requires;
   - the action when the condition holds and when it does not;
   - any time limit;
   - the document scope (the project, agency or network the clause governs);
   - the governing words or locator (`answer_basis`).

   This eligibility review checks that each inventoried row states a measured quantity with a threshold, tolerance or time limit and an action. Rows the author marks `meets_clause_rule = no` are reported as inventory rows that do not meet the clause rule, with the author's reasons. They are excluded from scoring and from the denominator of the share expressible unchanged. The paper reports the 564 total inventory rows and the number excluded. This review is added before the author supplies any answers.

   The author also marks `interpretation_needed` (yes or no, with a reason). Clauses marked yes are listed and counted but not scored.
2. **Derivation.** A deterministic script, `derive_answers.py`, turns each completed form into expected answers for that clause's fixtures. It skips answer derivation for rows marked `meets_clause_rule = no` and preserves their exclusion reasons in the review sheet. It reads only the form and the fixture inputs. It does not import, call or reuse any part of the evaluator, its schema or its operators. Its tests are written from this amendment's form definition alone. The script and its tests are committed with this amendment, before the author starts.
3. **Review and override.** The derived answers for each clause are shown to the author beside the completed form, before the answer file is hashed. The author confirms them, or overrides any of them by direct entry. Where the form cannot determine a fixture's answer, the script returns "form insufficient" and the author answers that fixture directly.
4. **Reported counts:**
   - all 564 inventory rows and the number marked `meets_clause_rule = no`, with their reasons;
   - forms completed;
   - clauses marked `interpretation_needed`;
   - fixtures answered by derivation;
   - fixtures answered directly because the form was insufficient;
   - derived answers the author overrode.
5. **Sealing.** The completed forms and the final answer file (derived answers plus direct entries and overrides) are hashed, the hashes committed, and tag `post-reg-revision-answers` created, all before the evaluator runs on any new fixture. The paper reports that an author completed the forms and confirmed the answers.

## Reporting groups

The Group A and Group B split of the original R5 is replaced, for reporting only, by the provenance of each expected answer:
- **Worked example:** the source states the outcome for a stated case.
- **Rule text:** the outcome follows from the clause wording.

Agreement is reported overall and for each provenance.

## Unchanged

- The four outcome categories per clause.
- The headline: the share expressible unchanged among inventory rows meeting the clause rule, with its count. The paper also reports the 564 total inventory rows and the number excluded by the author.
- The comparison with the Shapes Constraint Language (R6).
- Every consequence for the text stated in R5.
