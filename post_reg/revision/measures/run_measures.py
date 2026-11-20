"""Run the committed clause measures with the frozen evaluator on a CPU pod."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from text_checks import consistency, measurable, physical


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        path.write_text("clause_id\n")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_inputs(pr, passages):
    seals = json.loads((HERE / "input_seals.json").read_text())
    for relative, expected in seals["repository"].items():
        path = pr / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Frozen clause input changed: " + relative)
    if sha(passages) != seals["private_passages_sha256"]:
        raise ValueError("Private source passage checksum differs")


def classify(requirement, has_predicate):
    op = requirement["operator"]
    if op in {"measure", "maximum", "target_band", "paired_equivalence", "recall_lower_bound",
              "contract_repair_window", "contract_crack_trigger", "scanner_rci_class"}:
        return "expressible_unchanged", ""
    if requirement.get("extension_operator") and has_predicate:
        return "requires_new_operator", requirement["extension_operator"]
    return "not_representable", "An executable comparison with the printed limit is unavailable"


def normalized_action(raw):
    return {"insufficient_measurement": "cannot assess", "unit_error": "unit mismatch"}.get(raw, raw)


def measure_fixture(row, evaluate, is_measurable, physical_quantity):
    record = json.loads(row["input_record_json"])
    result = evaluate(record)
    raw = result["action"]
    return dict(fixture_id=row["fixture_id"], clause_id=row["clause_id"], measurable=is_measurable,
                fixture_kind="box_only" if row["fixture_id"].endswith("--box-only") else
                             "unit_mismatch" if row["fixture_id"].endswith("--units") else
                             "worked_example" if "--route-aaa-" in row["fixture_id"] else "boundary",
                physical_quantity=physical_quantity, raw_action=raw, action=normalized_action(raw),
                raw_result=json.dumps(result, default=str, sort_keys=True))


def verify_inputs(pr, passages):
    seals = json.loads((HERE / "input_seals.json").read_text())
    for relative, expected in seals["files"].items():
        path = pr / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Sealed clause input differs: " + relative)
    if hashlib.sha256(passages.read_bytes()).hexdigest() != seals["passages_sha256"]:
        raise ValueError("Sealed source passages differ")


def classify(requirement, has_predicate):
    original = {"measure", "maximum", "target_band", "paired_equivalence", "recall_lower_bound",
                "contract_repair_window", "contract_crack_trigger", "scanner_rci_class"}
    if requirement["operator"] in original:
        return "expressible_unchanged", ""
    if requirement.get("extension_operator") and has_predicate:
        return "requires_new_operator", requirement["extension_operator"]
    return "not_representable", "No executable source predicate in the frozen record or committed shape"


def add_graph_value(graph, node, name, value, ns):
    from rdflib import BNode, Literal
    if isinstance(value, list):
        for item in value:
            add_graph_value(graph, node, name, item, ns)
    elif isinstance(value, dict):
        child = BNode()
        graph.add((node, ns[name], child))
        for key, item in value.items():
            add_graph_value(graph, child, key, item, ns)
    elif value is not None:
        graph.add((node, ns[name], Literal(value)))


def validate_shape(task):
    identifier, clause, evidence, shape_path, target = task
    from pyshacl import validate
    from rdflib import Graph, Namespace
    from rdflib.namespace import RDF
    ns = Namespace("urn:qualification:revision:")
    graph = Graph()
    node = ns[identifier]
    graph.add((node, RDF.type, ns[target]))
    for key, value in evidence.items():
        add_graph_value(graph, node, key, value, ns)
    ok, report, _ = validate(graph, shacl_graph=shape_path, use_shapes=[str(ns[clause + "-shape"])])
    return dict(fixture_id=identifier, clause_id=clause, shacl_conforms=bool(ok),
                data_triples=len(graph), validation_triples=len(report))


def summarize(inventory, eligibility, coverage, outcomes, consistency_rows, published):
    result = []
    for source in ["all", *sorted({row["source_id"] for row in inventory})]:
        ids = {row["clause_id"] for row in inventory if source == "all" or row["source_id"] == source}
        measurable_ids = {key for key in ids if eligibility[key]}
        for label in ("expressible_unchanged", "requires_new_operator", "not_representable"):
            value = sum(row["clause_id"] in measurable_ids and row["classification"] == label for row in coverage)
            result.append(dict(measure="M1", source_id=source, category=label, n=value,
                               N=len(measurable_ids), share=value / len(measurable_ids) if measurable_ids else ""))
        result.append(dict(measure="M0", source_id=source, category="not_measurable",
                           n=len(ids - measurable_ids), N=len(ids)))
    for label, eligible in (
        ("M2", [row for row in outcomes if row["measurable"] and row["physical_quantity"] and row["fixture_kind"] == "box_only"]),
        ("M4", [row for row in outcomes if row["fixture_kind"] == "unit_mismatch"]),
    ):
        expected = "cannot assess" if label == "M2" else "unit mismatch"
        n = sum(row["action"] == expected for row in eligible)
        result.append(dict(measure=label, source_id="all", n=n, N=len(eligible),
                           share=n / len(eligible) if eligible else ""))
    eligible = [row for row in consistency_rows if eligibility[row["clause_id"]] and row["encoded"]]
    n = sum(row["passed"] for row in eligible)
    result.append(dict(measure="M3", source_id="all", n=n, N=len(eligible), share=n / len(eligible) if eligible else ""))
    for source in sorted({row["source_id"] for row in published}):
        entries = [row for row in published if row["source_id"] == source]
        result.append(dict(measure="M5", source_id=source, n=sum(row["agreement"] for row in entries), N=len(entries)))
    return result


def run(args):
    from concurrent.futures import ProcessPoolExecutor
    from rdflib import Graph, Namespace
    from rdflib.namespace import RDF
    pr = HERE.parents[1]
    verify_inputs(pr, args.passages)
    sys.path.insert(0, str(pr / "scripts"))
    sys.path.insert(0, str(pr / "supplements/qualification"))
    from conformance import fhwa, verdict
    from transfer import cases, extended
    clauses = pr / "revision/clauses"
    inventory = read_csv(clauses / "clause_inventory.csv")
    fixtures = read_csv(clauses / "fixtures.csv")
    passages = json.loads(args.passages.read_text())
    if set(passages) != {row["clause_id"] for row in inventory}:
        raise ValueError("Source passages do not cover the frozen inventory")
    args.out.mkdir(parents=True, exist_ok=False)
    shapes_file = clauses / "shapes.ttl"
    shapes = Graph().parse(shapes_file)
    ns = Namespace("urn:qualification:revision:")
    sh = Namespace("http://www.w3.org/ns/shacl#")
    by_clause = {}
    for row in fixtures:
        by_clause.setdefault(row["clause_id"], []).append(row)

    eligibility, physical_rows = {}, {}
    m0, m1, m3 = [], [], []
    for row in inventory:
        key = row["clause_id"]
        source = passages[key]["text"]
        eligibility[key] = measurable(source)
        physical_rows[key] = physical(row["quantity"], row["unit"])
        m0.append(dict(clause_id=key, source_id=row["source_id"], locator=row["locator"],
                       measurable=eligibility[key], passage_sha256=hashlib.sha256(source.encode()).hexdigest()))
        req = json.loads(by_clause[key][0]["input_record_json"])["requirement"]
        shape_node = ns[key + "-shape"]
        has_predicate = any(shapes.objects(shape_node, sh.sparql)) or any(shapes.objects(shape_node, sh["or"]))
        category, missing = classify(req, has_predicate)
        m1.append(dict(clause_id=key, source_id=row["source_id"], measurable=eligibility[key],
                       classification=category, missing_capability=missing, operator=req["operator"],
                       encoding=json.dumps(req, sort_keys=True)))
        record = consistency(req, source)
        m3.append(dict(clause_id=key, source_id=row["source_id"], encoded=category != "not_representable",
                       **record, locator=row["locator"],
                       source_text=source if not record["passed"] else ""))

    outputs = [measure_fixture(row, extended, eligibility[row["clause_id"]], physical_rows[row["clause_id"]])
               for row in fixtures]
    actual = {row["fixture_id"]:row for row in outputs}
    published = []
    for row in read_csv(HERE / "published_outcomes.csv"):
        got = actual[row["fixture_id"]]["action"]
        published.append(dict(**row, actual=got, agreement=got == row["published_action"]))
    vendor_count = 0
    for rec, key in fhwa():
        if rec["requirement"]["id"] == "FHWA-B-P":
            continue
        got = verdict(rec)["action"]
        published.append(dict(fixture_id=rec["requirement"]["id"], source_id="registered_vendor_guideline",
                              phase="development", published_action=key["decision"], actual=got,
                              agreement=got == key["decision"]))
        vendor_count += 1
    if vendor_count != 7 or len(published) != 30:
        raise ValueError("Published-outcome case coverage differs")

    tasks = [(row["fixture_id"], row["clause_id"], json.loads(row["input_record_json"])["evidence"],
              str(shapes_file), row["clause_id"]) for row in fixtures]
    registered = list(cases())
    if len(registered) != 12:
        raise ValueError("Registered transfer case count differs")
    for rec, _, _ in registered:
        key = "registered-" + rec["requirement"]["id"].replace(".", "-")
        tasks.append((rec["requirement"]["id"], key, rec["evidence"], str(shapes_file), key))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        shape_outputs = list(pool.map(validate_shape, tasks, chunksize=8))
    summary = summarize(inventory, eligibility, m1, outputs, m3, published)
    write_csv(args.out / "m0_measurability.csv", m0)
    write_csv(args.out / "m1_coverage.csv", m1)
    write_csv(args.out / "m3_source_consistency_private.csv", m3)
    write_csv(args.out / "fixture_outputs.csv", outputs)
    write_csv(args.out / "published_outcomes.csv", published)
    write_csv(args.out / "shacl_outputs.csv", shape_outputs)
    write_csv(args.out / "measures_summary.csv", summary)
    exceptions = [row | {"measure":"M2"} for row in outputs if row["measurable"] and row["physical_quantity"]
                  and row["fixture_kind"] == "box_only" and row["action"] != "cannot assess"]
    exceptions += [row | {"measure":"M4"} for row in outputs if row["fixture_kind"] == "unit_mismatch"
                   and row["action"] != "unit mismatch"]
    write_csv(args.out / "refusal_exceptions.csv", exceptions)
    boundaries = [row for row in outputs if row["fixture_kind"] == "boundary"]
    write_csv(args.out / "boundary_outputs.csv", boundaries)
    capabilities = []
    for row in m1:
        caps = [item for item in shape_outputs if item["clause_id"] == row["clause_id"]]
        fields = len(json.loads(row["encoding"]))
        capabilities.append(dict(clause_id=row["clause_id"], frozen_category=row["classification"],
                                 record_fields=fields, shape_count=1,
                                 shape_present=(ns[row["clause_id"] + "-shape"], RDF.type, sh.NodeShape) in shapes,
                                 shape_numeric_constraint=any(shapes.objects(ns[row["clause_id"]+"-shape"],sh.sparql)),
                                 missing_evidence="Property violations",
                                 supplied_lower_bound_comparison="Separate supplied-recall-bound-shape",
                                 interval_estimation="External calculation",
                                 validated_fixtures=len(caps)))
    write_csv(args.out / "r6_capabilities.csv", capabilities)
    report = ["# Clause measures", "", "These are the retrospective measures specified in approved amendment 3.", "",
              "The source check uses the cited printed-page ranges, or the numbered subsection for the restricted standard. "
              "Matching values and units in that text establishes the stated mechanical consistency criterion.", "",
              "The boundary outputs are inspectable executions. Agreement is evaluated only for the published cases.", "",
              "The seven vendor verdicts are development cases.", ""]
    for row in summary:
        report.append(f"- {row['measure']} {row['source_id']} {row.get('category','')}. {row['n']} of {row['N']}.")
    report += ["", "M3 failure text includes restricted source passages and remains private.",
               "SHACL compares supplied evidence to constraints. Interval estimation remains an external calculation."]
    (args.out / "report").mkdir()
    (args.out / "report/REVISION_RESULTS.md").write_text("\n".join(report) + "\n")
    print("Clause measures completed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args)


if __name__ == "__main__":
    main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args)


if __name__ == "__main__":
    main()
