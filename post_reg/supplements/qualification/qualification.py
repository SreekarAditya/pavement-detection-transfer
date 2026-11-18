"""Evaluators for the version 1 qualification record."""
import math
from statistics import mean, stdev

SCHEMA = "qualification/1"


def decision(action, reason, **fields):
    return dict(action=action, reason=reason, **fields)


def evaluate(requirement, evidence):
    op = requirement["operator"]
    if op not in {"paired_equivalence", "recall_lower_bound", "maximum", "target_band", "measure"}:
        return decision("unrepresentable", f"Unsupported operator {op}")
    missing = [name for name in requirement["required_evidence"] if name not in evidence]
    if missing:
        return decision("insufficient_measurement", "Required evidence is missing", missing=missing)
    if evidence.get("unit") != requirement["unit"]:
        return decision("unit_error", "Units differ and no conversion was declared")
    if op == "paired_equivalence":
        from scipy.stats import t
        pairs = evidence["paired_subsections"]
        differences = [row["vendor"] - row["reference"] for row in pairs]
        n = len(differences)
        if n < 2:
            return decision("insufficient_measurement", "At least two paired subsections are required")
        estimate = mean(differences)
        se = stdev(differences) / math.sqrt(n)
        lo, hi = requirement["lower"], requirement["upper"]
        if se <= 0:
            return decision("insufficient_measurement", "The paired t test requires nonzero variation")
        tlo, thi = (estimate - lo) / se, (estimate - hi) / se
        p_lo, p_hi = float(t.sf(tlo, n - 1)), float(t.cdf(thi, n - 1))
        p = max(p_lo, p_hi)
        return decision("equivalent" if p < requirement["alpha"] else "equivalence_not_demonstrated",
                        "Paired two one-sided t tests", n=n, difference=estimate, difference_sd=stdev(differences),
                        t_lower=tlo, t_upper=thi, p_lower=p_lo, p_upper=p_hi, p_governing=p,
                        vendor_mean=mean(row["vendor"] for row in pairs),
                        reference_mean=mean(row["reference"] for row in pairs),
                        vendor_sd=stdev(row["vendor"] for row in pairs),
                        reference_sd=stdev(row["reference"] for row in pairs), difference_se=se,
                        vendor_se=stdev(row["vendor"] for row in pairs) / math.sqrt(n),
                        reference_se=stdev(row["reference"] for row in pairs) / math.sqrt(n))
    if op == "recall_lower_bound":
        from scipy.stats import norm
        from pr_common import ratio_ci
        blocks = evidence["audit_blocks"]
        p, lo, hi, se = ratio_ci([b["matched"] for b in blocks], [b["refs"] for b in blocks],
                                  norm.ppf(.5 + requirement["level"] / 2))
        return decision("accept" if math.isfinite(lo) and lo >= requirement["floor"] else "adapt",
                        "Frozen lower-bound decision" if math.isfinite(lo) else "Audit bound unavailable",
                        estimate=float(p), lower=float(lo), upper=float(hi), se=float(se))
    if op == "maximum":
        return decision("within_maximum" if evidence["value"] <= requirement["upper"] else "above_maximum",
                        "Compare observed rate with its maximum")
    if op == "target_band":
        x = evidence["value"]
        result = "below_target" if x < requirement["lower"] else "above_target" if x > requirement["upper"] else "within_target"
        return decision(result, "Compare observed rate with its target band")
    return decision("measurement_present", "Required measurement fields are present")
