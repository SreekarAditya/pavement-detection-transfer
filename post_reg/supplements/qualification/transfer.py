"""Apply the frozen US evaluator first, then the specification transfer extension."""
from pathlib import Path
import sys

from conformance import fhwa, record, save_csv, specifications, verdict, write_json
from qualification import decision

MORTH="https://morth.gov.in/sites/default/files/comprehensive_compendium_circular/2-CCC-130.248-%20AE%20RFP.pdf"
NH91="https://nhai.gov.in/nhai/sites/default/files/tender/OtherDocuments/4_Non_Technical_Schedules.pdf"
NH27="https://nhai.gov.in/nhai/sites/default/files/2020/Schedule_0.pdf"
DFT="https://www.gov.uk/government/publications/road-network-size-and-condition-statistics-guidance/road-conditions-notes-and-definitions"
SCHEDULES={"NH91-Aligarh-Bhadwas": (48,NH91,"Schedule K, Annex I, iii, PDF p. 24"),
           "NH27-Lucknow-Ayodhya": (24,NH27,"Schedule E, Annex I, iii, PDF p. 191")}


def cases():
    base=dict(unit="physical_dimensions", image_box=[20,30,80,90])
    for id_, name, needed in [("morth-pothole","pothole",["area_m2","depth_mm"]),
                              ("morth-crack","crack",["length_m","width_mm","depth_mm"])]:
        for complete in (False, True):
            rec=record(id_+("-measurements" if complete else "-box"),"measure",
                       name+" physical dimensions","physical_dimensions","one defect","Authority Engineer survey",
                       needed, {**base, **({k: dict(value=1.0, provenance="synthetic physical measurement") for k in needed}
                                          if complete else {})},MORTH,"§6.4.2, printed p. 114")
            yield rec,"measurement_present" if complete else "insufficient_measurement",None
    for contract in (*SCHEDULES,None):
        hours,source,locator=SCHEDULES[contract] if contract else (None,"Project agreement required","")
        yield record(f"pothole-{contract or 'country-only'}","contract_repair_window","confirmed pothole",
                     "defect_present","one defect","carriageway or paved shoulder",["confirmation"],
                     dict(unit="defect_present", confirmation=True, agreement=contract, country="India"),
                     source,locator), "repair_window" if contract else "ambiguous_scope",hours
    for share in (5.0,5.1):
        yield record(f"nh91-cracking-{share}","contract_crack_trigger","physical cracked road area share",
                     "percent","1 km stretch","NH-91 carriageway or paved shoulder",["value","segment_km"],
                     dict(unit="percent", value=share, segment_km=1.0, agreement="NH91-Aligarh-Bhadwas"),
                     NH91,"Schedule K, Annex I, iv, PDF p. 24",lower=5.0), "repair_window" if share>5 else "outside_clause",720 if share>5 else None
    for score in (99,100,None):
        yield record(f"scanner-{score}","scanner_rci_class","Road Condition Indicator",
                     "rci_score","road section","English SCANNER scheme",["score"],
                     dict(unit="rci_score", score=score) if score is not None else dict(unit="count",boxes=2),
                     DFT,"SCANNER RCI notes, Green/Amber/Red"), (
                         "insufficient_measurement" if score is None else "amber" if score<100 else "red"),None


def extended(rec):
    req,e=rec["requirement"],rec["evidence"]
    op=req["operator"]
    if op not in {"contract_repair_window","contract_crack_trigger","scanner_rci_class"}:
        return verdict(rec)
    missing=[k for k in req["required_evidence"] if k not in e or e[k] is None]
    if missing:
        return decision("insufficient_measurement","Required evidence is missing",missing=missing)
    if e.get("unit")!=req["unit"]:
        return decision("unit_error","Units differ")
    if op=="scanner_rci_class":
        s=e["score"]
        if not 0<=s<=315:
            return decision("invalid_measurement","RCI is outside the reported range")
        name="green" if s<=40 else "amber" if s<100 else "red"
        return decision(name,"Classification of the supplied SCANNER RCI")
    if e.get("agreement") not in SCHEDULES:
        return decision("ambiguous_scope","The applicable project agreement is required")
    if op=="contract_repair_window":
        if e["confirmation"] is not True:
            return decision("insufficient_measurement","Pothole confirmation is required")
        return decision("repair_window","Allowance under the identified agreement",
                        allowance_hours=SCHEDULES[e["agreement"]][0])
    if e["agreement"]!="NH91-Aligarh-Bhadwas" or e["segment_km"]!=1:
        return decision("ambiguous_scope","The clause requires the named agreement and a 1 km stretch")
    if e["value"]<=req["lower"]:
        return decision("outside_clause","Prevalence does not exceed the strict scope threshold")
    return decision("repair_window","Cracking exceeds the strict 5% threshold",allowance_hours=720)


def run(out):
    out.mkdir(parents=True)
    rows, records, answers = [],[],[]
    for rec, expected,hours in cases():
        before,after=verdict(rec),extended(rec)
        r=rec["requirement"]
        rows.append({k:r[k] for k in ("id","source","locator","measurement","unit","observation_unit",
                                     "operator","reference_population","required_evidence")})
        rows[-1].update(baseline=before["action"], baseline_reason=before["reason"],
                        after=after["action"], after_reason=after["reason"], missing=after.get("missing",[]),
                        allowance_hours=after.get("allowance_hours"), expected=expected, expected_hours=hours,
                        baseline_match=before["action"]==expected and before.get("allowance_hours")==hours,
                        after_match=after["action"]==expected and after.get("allowance_hours")==hours,
                        added="contract identity and scoped allowance/trigger" if r["operator"].startswith("contract")
                        else "RCI range classification" if r["operator"].startswith("scanner") else "")
        records.append(rec)
        answers.append(dict(id=r["id"], action=expected, allowance_hours=hours))
    assert all(extended(rec)["action"] == expected for rec,expected in specifications())
    assert all(extended(rec) == verdict(rec) for rec,_ in fhwa())
    for rec in records:
        rec["schema_version"]="qualification/1.1"
    save_csv(out/"transfer_cases.csv",rows)
    write_json(out/"fixture_inputs.json",records)
    write_json(out/"fixture_answer_key.json",answers)
    (out/"REPORT.md").write_text(
        "# Task 1A · Transfer across engineering specifications\n\n"
        f"The frozen US evaluator reproduced {sum(r['baseline_match'] for r in rows)} of {len(rows)} "
        f"rule-derived fixtures. The extension reproduced {sum(r['after_match'] for r in rows)} of {len(rows)}.\n\n"
        "The existing measurement operator expresses the MoRTH field requirements. Contract-specific allowances "
        "and triggers require the governing agreement. An absolute deadline additionally requires a clock-start event. "
        "SCANNER classification requires the edition and an observed Road Condition Indicator score.\n\n"
        "The cited 24-hour source is the July 2021 Lucknow–Ayodhya NH-27 draft agreement. The 48-hour source is "
        "the NH-91 Aligarh–Bhadwas bid schedule. The fixtures test representational coverage of selected clauses.\n")


if __name__=="__main__":
    run(Path(sys.argv[1]))
