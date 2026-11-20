# Clause forms and answer review

Complete one row per clause in the private copy of the forms. Use the cited source to determine the rule. The excerpt is a reading aid. Use the original clause and its context for your answer basis.

| Field | Entry |
| --- | --- |
| `meets_clause_rule`, `clause_rule_reason` | Mark `yes` or `no` and give your reason. A `no` row is retained with its reason, skipped in derivation and excluded from scoring and the headline denominator. The report gives the 564 total and the excluded count. |
| `measured_quantity`, `unit` | Quantity and exact observation unit. Read the fixture observations in the review sheet for the recorded units. |
| `denominator_or_aggregation` | The denominator, aggregation or observation scope stated by the source. Enter the scope even when one observation suffices. |
| `evidence_field`, `required_evidence` | The name of the observation to compare and a comma-separated list of all required observation names. Dots address nested fields. |
| `comparison`, `threshold` | Use the comparison formats below. Use ISO dates or `MM-DD` for calendar thresholds. |
| `lower_included`, `upper_included` | Enter `yes` or `no` at the endpoint of an interval. The scalar symbols already specify inclusion. |
| `action_when_true`, `action_when_false` | The two source-based outcomes in your own words. |
| `time_limit` | The source's time allowance where applicable, including the unit. An action can include `{time_limit}` to insert it. |
| `document_scope`, `answer_basis` | Governing program, project or network and the source locator or governing words. |
| `interpretation_needed`, `interpretation_reason` | Mark `yes` and give a reason when answering requires interpretation beyond the source. Mark `no` otherwise. |

Scalar comparisons accept `<`, `<=`, `>`, `>=`, `=` and `!=`. Use a single threshold such as `24`.

Use `between` or `outside` with a JSON pair such as `[10,20]` and both inclusion fields. Use `in` or `not_in` with a JSON list such as `[1,3,5]`.

Use `all` or `any` with a JSON list of conditions. Each condition has its own `evidence_field`, `comparison` and `threshold`. Add the inclusion fields to a condition if it is an interval. The `at_least` comparison takes a JSON object with an integer `minimum` and a `conditions` list. List the observations required by these conditions in the form's `required_evidence`.

Use `deadline` with a time limit such as `24 hour` and set `upper_included`. The time unit must exactly match the observation unit. Leave `threshold` blank for this comparison.

Read the completed forms beside the derived answers in a fresh review sheet. Confirm the derived answer or supply an override. Enter a direct answer where the sheet reports `form insufficient`. A box-only observation that lacks required measurements gives `cannot assess`. A mismatched unit is reported as `unit mismatch`. An interpretation-needed case is excluded from scoring.

Run the script with the private completed forms and private fixture CSV. Choose a new output filename so an existing review with author edits is protected.

```bash
python3 post_reg/revision/derive_answers.py \
  --forms "<private-folder>/clause_forms.csv" \
  --fixtures "<private-folder>/fixtures_full.csv" \
  --out "<private-folder>/clause_answer_review_completed.csv"
```
