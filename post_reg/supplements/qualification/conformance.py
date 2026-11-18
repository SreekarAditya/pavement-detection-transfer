"""Run the frozen engineering-record conformance cases and record exclusions."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from qualification import SCHEMA, evaluate

DATA = Path(__file__).with_name("data")
FHWA = "https://rosap.ntl.bts.gov/view/dot/55791/dot_55791_DS1.pdf"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def record(id_, operator, measurement, unit, observation_unit, population, required, evidence,
           source="Synthetic specification fixture", locator="", uncertainty="None", **parameters):
    requirement = dict(id=id_, source=source, locator=locator, measurement=measurement, unit=unit,
                       observation_unit=observation_unit, reference_population=population,
                       required_evidence=required, operator=operator, uncertainty=uncertainty, **parameters)
    return dict(schema_version=SCHEMA, requirement=requirement, evidence=evidence)


def verdict(rec):
    return evaluate(rec["requirement"], rec["evidence"])


def fhwa():
    raw = read_csv(DATA / "fhwa_measurements.csv")
    for key in read_csv(DATA / "fhwa_answer_key.csv"):
        s, v = key["site"], key["vendor"]
        group = [r for r in raw if r["site"] == s]
        pairs = [dict(id=r["subsection"], reference=float(r["reference_percent"]),
                      vendor=float(r[f"vendor_{v.lower()}_percent"])) for r in group]
        obs = dict(unit="percent", paired_subsections=pairs)
        rec = record(f"FHWA-{s}-{v}", "paired_equivalence", "HPMS cracking percent", "percent",
                     "paired roadway subsection", f"Reference site {s}", ["paired_subsections"], obs, FHWA,
                     f"{group[0]['locator']}; Figure {key['figure']}", "paired t equivalence",
                     lower=float(key["lower"]), upper=float(key["upper"]), alpha=float(key["alpha"]),
                     tolerance_unit="percentage_point")
        yield rec, key


def specifications():
    yield record("wheel-path", "measure", "wheel-path crack length", "m", "road subsection", "illustrative",
                 ["physical_length"], {"unit": "pixels", "boxes": [1]}), "insufficient_measurement"
    yield record("incompatible-unit", "measure", "cracking percentage-point difference", "percentage_point",
                 "subsection", "illustrative", ["value"], dict(unit="fraction", value=.09)), "unit_error"
    for op, expected, limits in (("maximum", "within_maximum", dict(upper=150)),
                                 ("target_band", "below_target", dict(lower=135, upper=165))):
        yield record(f"alert-{op}", op, "alerts per 100 images", "alerts_per_100_images", "image", "illustrative",
                     ["value"], dict(unit="alerts_per_100_images", value=90), **limits), expected


def detector_inputs(audit):
    counts = defaultdict(list)
    for row in read_csv(audit / "audit_blocks.csv"):
        counts[row["run_id"]].append({k: int(row[k]) for k in ("refs", "matched")})
    for expected in read_csv(audit / "fits.csv"):
        rid = expected["run_id"]
        evidence = dict(unit="fraction", audit_blocks=counts[rid])
        yield record(rid, "recall_lower_bound", "mapped reference-box recall", "fraction",
                     "audit block", expected["collection"], ["audit_blocks"], evidence,
                     "Verified retrospective nine-fit audit", f"audit_blocks.csv:{rid}",
                     "frozen block ratio linearization", floor=.30, level=.90), expected["interval_action"]


def write_json(path, data):
    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, list):
            return [clean(y) for y in x]
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        return x
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def run(audit, out):
    out.mkdir(parents=True)
    worked, decisions, differences = [], [], []
    for rec, key in fhwa():
        rid = rec["requirement"]["id"]
        result = verdict(rec)
        phase = "development" if rid == "FHWA-B-P" else "worked_comparison"
        decisions.append(dict(id=rid, group="FHWA", phase=phase, expected=key["decision"], actual=result["action"],
                              agreement=result["action"] == key["decision"]))
        for field in ("n","difference","difference_sd","vendor_mean","vendor_sd","vendor_se","reference_mean",
                      "reference_sd","reference_se","t_lower","p_lower","t_upper","p_upper","p_governing"):
            printed = key[field]
            decimals = len(printed.split(".", 1)[1]) if "." in printed else 0
            comparison = float(printed)
            actual = result[field]
            differences.append(dict(id=rid, figure=key["figure"], field=field, printed=printed, computed=actual,
                                    difference=actual-comparison,
                                    agrees_at_printed_precision=abs(actual-comparison) <= 0.5 * 10**(-decimals)+1e-12,
                                    transcription_note=key["note"]))
        worked.append({**rec, "decision": result})
    for group, cases in [("Detector", detector_inputs(audit)), ("Specification", specifications())]:
        for rec, expected in cases:
            result = verdict(rec)
            decisions.append(dict(id=rec["requirement"]["id"], group=group, expected=expected,
                                  actual=result["action"], agreement=result["action"] == expected))
            worked.append({**rec, "decision": result})
    tests = [r for r in decisions if r["group"] == "Specification"]
    assert len(tests) == 4 and len([r for r in decisions if r["group"] == "Detector"]) == 9
    save_csv(out / "conformance.csv", decisions)
    save_csv(out / "printed_precision.csv", differences)
    save_csv(out / "exclusions.csv", read_csv(DATA / "fhwa_excluded.csv"))
    for name in ("fhwa_measurements.csv", "fhwa_answer_key.csv"):
        (out / name).write_bytes((DATA / name).read_bytes())
    write_json(out / "worked_records.json", worked)
    write_json(out / "blank_record.json", record("", "measure", "", "", "", "", [], {}))
    stats = {}
    for group in ("FHWA", "Detector", "Specification"):
        subset = [d for d in decisions if d["group"]==group and d.get("phase")!="development"]
        positive = {"FHWA": "equivalent", "Detector": "accept"}.get(group)
        stats[group] = dict(total=len(subset), matching=sum(d["agreement"] for d in subset),
                            always_accept=sum(d["expected"]==positive for d in subset) if positive else None,
                            always_decline=sum(d["expected"]!=positive for d in subset) if positive else None)
    write_json(out / "summary.json", stats)
    row = decisions[0]
    lines = ["# Task 1 · Engineering qualification conformance", "",
             "The FHWA cases use paired roadway subsections. The retrospective detector reconstruction uses audit-block reference boxes. The published comparator keys are stored separately.", "",
             f"Development case {row['id']} matched the published verdict: {row['agreement']}.", "",
             "| Group | Verdicts matched | Always accept | Always decline |",
             "| --- | ---: | ---: | ---: |"]
    for name, stat in stats.items():
        lines.append(f"| {name} | {stat['matching']}/{stat['total']} | {stat['always_accept'] if stat['always_accept'] is not None else '—'} | {stat['always_decline'] if stat['always_decline'] is not None else '—'} |")
    lines.extend(["", "The specification group contains three cases, with two operator outputs for the rate case.",
                  "The FHWA comparison excludes ten Table 7 site–vendor decisions without paired subsections.",
                  "The precision table preserves the differences from the displayed values. Figure 22 labels its descriptive vendor row P, while its title and paired test identify Q."])
    (out / "REPORT.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("audit_results", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.audit_results, a.output)
