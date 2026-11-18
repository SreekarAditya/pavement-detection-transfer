"""Shared calibration-only rules for the retrospective and independent evaluations."""
import numpy as np
from scipy.stats import norm

from pr_common import ratio_ci

BUDGETS = (100, 200)
Z90, Z95 = norm.ppf([.95, .975])
ARMS = ("point_100", "point_200", "lower_bound_100", "lower_bound_200",
        "full_90", "full_95", "selective_95")


def ordered_blocks(blocks, seed):
    return np.random.default_rng(np.random.SeedSequence(seed)).permutation(blocks)


def look(order, images, matched, refs, target):
    c = np.cumsum(images[order])
    n = min(len(order), np.searchsorted(c, target) + 1)
    prefix = order[:n]
    boxes = int(refs[prefix].sum())
    actual = int(images[prefix].sum())
    p, lo, hi, se = ratio_ci(matched[prefix], refs[prefix], Z90)
    valid = actual >= target and n >= 2 and boxes > 0 and np.isfinite(se) and se > 0
    reason = ("shortfall" if actual < target else "too_few_blocks" if n < 2 else
              "no_references" if boxes == 0 else "nonpositive_or_nonfinite_se" if not valid else "")
    return dict(images=actual, boxes=boxes, blocks=int(n), target=target, recall=float(p), se=float(se),
                lo90=float(lo),hi90=float(hi),lo95=float(p-Z95*se),hi95=float(p+Z95*se),
                valid=bool(valid), evidence_reason=reason)


def three_way(view, level, floor):
    if not view["valid"]:
        return "unresolved", view["evidence_reason"]
    lo, hi = view[f"lo{level}"], view[f"hi{level}"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unresolved", "nonfinite_bounds"
    if lo >= floor:
        return "accept", "lower_bound_meets_floor"
    if hi < floor:
        return "adapt", "upper_bound_below_floor"
    return "unresolved", "interval_straddles_floor"


def policies(order, images, matched, refs, floor):
    first, full = (look(order, images, matched, refs, t) for t in BUDGETS)
    rows = {}
    def put(name, selected, action, reason, requested=False, second=False):
        rows[name] = dict(**selected, action=action, action_reason=reason,
                          second_requested=requested, second_audit=second,
                          no_distinct_second=full["blocks"] == first["blocks"])
    for view in (first, full):
        for name, value in (("point",view["recall"]), ("lower_bound",view["lo90"])):
            acceptable = np.isfinite(value) and value >= floor
            put(f"{name}_{view['target']}",view,"accept" if acceptable else "adapt",
                "estimate_meets_floor" if acceptable else "estimate_unavailable" if not np.isfinite(value)
                else "estimate_below_floor")
    for level in (90,95):
        put(f"full_{level}",full,*three_way(full,level,floor))
    a0, reason = three_way(first,95,floor)
    if a0 != "unresolved":
        put("selective_95",first,a0,reason)
    elif full["blocks"] > first["blocks"]:
        put("selective_95",full,*three_way(full,95,floor),requested=True,second=True)
    else:
        put("selective_95",first,"unresolved","no_distinct_second:"+reason,requested=True)
    return rows, first, full
