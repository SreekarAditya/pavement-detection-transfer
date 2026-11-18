"""Prospective recording-block eligibility and qualification for BharatPotHole v2."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import zipfile

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from conformance import save_csv, write_json
import external
import pr_common as PC

RELEASE = "BharatPotHole v2"
URL = "https://www.kaggle.com/datasets/surbhisaswatimohanty/bharatpothole/versions/2"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/surbhisaswatimohanty/bharatpothole?datasetVersionNumber=2"
SIZE = 802413909
TOP = "BharatPotHole/BharatPotHole/"
INDEX = 2
FORM = re.compile(
    r"(?P<part>train|test|valid)/images/"
    r"(?P<recording>\d{14}_\d{4}|signal-\d{4}-\d{2}-\d{2}-\d{6}(?:_\d{3})?)"
    r"_frame_(?P<frame>\d+)_jpg\.rf\.[a-f0-9]{32}\.jpg"
)
need = external.need


def frame_key(path):
    match = FORM.fullmatch(path.removeprefix(TOP))
    if match is None:
        return None
    return match["recording"], int(match["frame"])


def video_key(recording):
    if recording.startswith("signal-"):
        return recording.split("_", 1)[0]
    return recording


def label_boxes(data):
    rows = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        cls, *coordinates = line.split()
        need(cls == "0" and len(coordinates) == 4, "Unrecognized annotation row")
        box = np.array([float(x) for x in coordinates])
        need(np.isfinite(box).all() and ((box >= 0) & (box <= 1)).all()
             and (box[2:] > 0).all(), "Invalid normalized pothole box")
        x1, y1, x2, y2 = external.clip_box(*box, 640)
        if x2 - x1 >= 1 and y2 - y1 >= 1:
            rows.append((x1, y1, x2, y2))
    return sorted(set(rows))


def label_path(member):
    return member.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"


def read_release(archive, images, out, G):
    families = defaultdict(list)
    ambiguous = []
    exact_edges = set()
    groups = []
    with zipfile.ZipFile(archive) as z:
        members = z.namelist()
        need(len(members) == len(set(members)), "Duplicate member paths")
        export = z.read(TOP + "README.roboflow.txt").decode()
        classes = yaml.safe_load(z.read(TOP + "data.yaml"))
        need(classes["nc"] == 1 and classes["names"] == ["pothole"], "Released class declaration differs")
        need("Resize to 640x640 (Stretch)" in export and "No image augmentation techniques were applied." in export,
             "Released preprocessing differs")
        source_paths = sorted(p for p in members if p.startswith(TOP) and "/images/" in p and not p.endswith("/"))
        for name in source_paths:
            key = frame_key(name)
            if key is None:
                ambiguous.append(name)
            else:
                families[key].append(name)
        need(len(source_paths) == 7074, "Released frame count differs from the version declaration")
        if ambiguous:
            save_csv(out / "unparsed_frame_paths.csv", [dict(member=name) for name in ambiguous])
        need(families, "No source recordings could be identified")
        items = sorted(families)
        exact = {}
        label_members = {n for n in members if "/labels/" in n and n.endswith(".txt")}
        for i, key in enumerate(items):
            members = families[key]
            for n, member in enumerate(members):
                data = z.read(member)
                digest = hashlib.sha256(data).digest()
                first = exact.setdefault(digest, i)
                if first != i:
                    exact_edges.add((first, i))
                if n == 0:
                    with Image.open(BytesIO(data)) as image:
                        need(image.size == (640, 640), "Unexpected exported frame dimensions")
                    name = Path(member).name
                    (images / name).write_bytes(data)
            groups.append(dict(recording=key[0], frame=key[1], video=video_key(key[0]),
                               representative=members[0], variants=len(members)))
        # Connected exact-byte duplicate families are represented by their first source frame.
        components = G.components(len(items), exact_edges)
        first = {}
        for i, component in enumerate(components):
            first.setdefault(component, i)
        representatives = sorted(first.values())
        usable = []
        records = []
        missing = empty = losing = 0
        for i in representatives:
            member = groups[i]["representative"]
            name = Path(member).name
            label = label_path(member)
            if label not in label_members:
                missing += 1
                continue
            data = z.read(label)
            if not data.strip():
                empty += 1
            boxes = label_boxes(data)
            losing += int(data.strip() != b"" and not boxes)
            records.extend((name, external.POTHOLE, *b) for b in boxes)
            usable.append((i, name))
        for i, group in enumerate(groups):
            original = first[components[i]]
            group["canonical_family"] = Path(groups[original]["representative"]).name
            group["status"] = "exact_duplicate" if original != i else (
                "paired" if label_path(group["representative"]) in label_members else "missing_annotation")
        save_csv(out / "frame_families.csv", groups)
        labels = pd.DataFrame(records, columns=["file", "cls", *PC.BOX])
        evidence = dict(released_frames=len(source_paths), ambiguous_frames=len(ambiguous),
                        original_frame_families=len(families), source_recording_ids=len({r for r, _ in items}),
                        video_ids_before_linking=len({x["video"] for x in groups}),
                        exact_duplicate_families=len(items) - len(representatives), missing_annotation=missing,
                        empty_annotation=empty, positive_annotations_lost_to_geometry=losing,
                        canonical_frames_with_known_annotation=len(usable))
        return usable, groups, exact_edges, labels, evidence


def screen(a):
    import torch
    PC.locked_scripts(a.root)
    import pdt_s2_g4 as G
    import pdt_c0 as C0
    torch.set_num_threads(a.workers)
    G.NH = G.NW = a.workers
    a.out.mkdir(parents=True, exist_ok=False)
    a.sources.mkdir(parents=True, exist_ok=True)
    images = a.sources / "bharat-images"
    images.mkdir()
    archive = a.sources / "bharat-v2.zip"
    if not archive.exists():
        temp = archive.with_suffix(".part")
        subprocess.run(["curl", "-fLsS", "--retry", "2", "--connect-timeout", "20", "-o", str(temp), DOWNLOAD],
                       check=True)
        temp.rename(archive)
    need(archive.stat().st_size == SIZE, "Version 2 archive size differs")
    archive_hash = PC.sha(archive)
    write_json(a.out / "source_receipt.json", dict(release=RELEASE, version=2, archive_sha256=archive_hash,
                bytes=SIZE, url=URL, adapter_sha256=PC.sha(Path(__file__))))
    print("Pinned source bytes. Verifying frame and annotation identities.", flush=True)
    usable, groups, exact, labels, evidence = read_release(archive, images, a.out, G)
    need(usable, "No usable frame and annotation pairs")
    print("Deduplication complete. Loading locked reference sets.", flush=True)
    names = [name for _, name in usable]
    paths = [str(images / n) for n in names]
    h = G.hash_table(paths, names)
    need(not h.phash.isna().any(), "Unreadable release images")
    ph = C0.hex_to_u64(h.phash)
    c0meta = json.loads((a.c0 / "c0_meta.json").read_text())
    model, weights = C0.load_sscd("cpu")
    need(model is not None and PC.sha(weights) == c0meta["descriptor_file_sha256"], "SSCD identity differs")
    rph, rd, rnames, provenance = external.reference_sets(a, G, C0, model)
    q = C0.embed(paths, model, "cpu", workers=a.workers)
    print("Comparing release frames to the locked references and linking recording groups.", flush=True)
    links = external.links(G, q, rd, ph, rph, G.LINK_COS)
    dropped = {i for i, _ in links}
    pd.DataFrame([(names[i], rnames[j]) for i, j in sorted(links)],
                 columns=["frame", "reference"]).to_csv(a.out / "overlap_removals.csv", index=False)
    layout = set(exact)
    first = {}
    for i, row in enumerate(groups):
        anchor = first.setdefault(row["video"], i)
        if i != anchor:
            layout.add((anchor, i))
    within = external.links(G, q, q, ph, ph, G.BLOCK_COS, same=True)
    layout.update((usable[i][0], usable[j][0]) for i, j in within)
    components = G.components(len(groups), layout)
    kept = [name for i, name in enumerate(names) if i not in dropped]
    keep = set(kept)
    blocks = {name: "video-" + str(components[i]) for i, name in usable if name in keep}
    labels = labels[labels.file.isin(kept)]
    positive = int(labels.file.nunique())
    box_count = len(labels)
    count = len(set(blocks.values()))
    gates = dict(video_groups=count >= 20, unique_positive_frames=positive >= 200, mapped_pothole_boxes=box_count >= 150)
    result = dict(release=RELEASE, source_identifier=URL, index=INDEX, archive_sha256=archive_hash,
                  adapter_sha256=PC.sha(Path(__file__)), descriptor_sha256=PC.sha(weights), **evidence, **provenance,
                  reference_images=len(rnames), overlap_removed_frames=len(dropped),
                  overlap_removed_india_frames=len({i for i,j in links if rnames[j].startswith("rdd2022_")
                                                     and "india" in rnames[j].lower()}),
                  independent_video_groups=count, unique_positive_frames=positive, mapped_pothole_boxes=box_count,
                  retained_frames=len(kept), cross_video_link_pairs=sum(groups[usable[i][0]]["video"] !=
                                                                      groups[usable[j][0]]["video"] for i,j in within),
                  largest_video_group=max(Counter(blocks.values()).values(), default=0),
                  gates=gates, status="eligible" if all(gates.values()) else "ineligible")
    pd.DataFrame(blocks.items(), columns=["file", "block"]).to_csv(a.out / "video_groups.csv", index=False)
    if all(gates.values()):
        split = PC.load_params(a.rules)["split"]
        cal, held = PC.make_split(kept, blocks, split["calibration_share"], split["seed"], split["balance"])
        need(cal and held and not set(cal) & set(held), "Empty or overlapping frame split")
        need(not {blocks[f] for f in cal} & {blocks[f] for f in held}, "A video group crosses the split")
        d = a.out / "split"
        d.mkdir()
        for name, keys in (("calibration", cal), ("heldout", held)):
            (d / f"{name}.txt").write_text("\n".join(keys) + "\n")
            f = d / ("calibration_labels.parquet" if name == "calibration" else "private/heldout_labels.parquet")
            f.parent.mkdir(exist_ok=True)
            labels[labels.file.isin(keys)].to_parquet(f, index=False)
            result[f"{name}_labels_sha256"] = PC.sha(f)
        save_csv(d / "blocks.csv", [dict(file=f, block=b) for f,b in blocks.items()])
        result["split_sha256"] = {p: PC.sha(d / p) for p in ("calibration.txt", "heldout.txt", "blocks.csv")}
    write_json(a.out / "eligibility.json", [result])
    (a.out / "REPORT.md").write_text(
        "# Task 3B · BharatPotHole v2 eligibility\n\n"
        f"Disposition: **{result['status']}**.\n\n"
        f"| Gate | Retained | Minimum | Pass |\n| --- | ---: | ---: | --- |\n"
        f"| Usable video groups | {count} | 20 | {gates['video_groups']} |\n"
        f"| Unique positive frames | {positive} | 200 | {gates['unique_positive_frames']} |\n"
        f"| Mapped pothole boxes | {box_count} | 150 | {gates['mapped_pothole_boxes']} |\n\n"
        "Recording identifiers were inferred from source filenames and connected for overlapping frames. "
        "Route independence beyond these checks is unverified. Workload is defined per 100 retained released frames. "
        "The release is a curated export with stretched geometry. Missing labels were excluded.\n\n"
        "The official export split was pooled. The release README declares CC BY 4.0 and no export augmentation. "
        "The Kaggle listing and code repository carry distinct license notices. Images are retained as private inputs.\n"
    )
    print(f"Eligibility: {result['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in ("screen", "commit", "score"):
        p = commands.add_parser(phase)
        for name in ("root", "rules", "out"):
            p.add_argument("--" + name, type=Path, required=True)
        if phase == "screen":
            for name in ("c0", "sources"):
                p.add_argument("--" + name, type=Path, required=True)
            p.add_argument("--workers", type=int, default=8)
        else:
            for name in ("split", "eligibility", "calibration", "heldout"):
                p.add_argument("--" + name, type=Path, required=True)
        if phase == "commit":
            for name in ("code_commit", "weight_sha"):
                p.add_argument("--" + name, required=True)
        if phase == "score":
            for name in ("commitment", "labels", "approved_repo"):
                p.add_argument("--" + name, type=Path, required=True)
            for name in ("commitment_sha", "approved_tag", "approved_path"):
                p.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.phase == "screen":
        screen(args)
        return
    args.identity = dict(candidate=RELEASE, source_identifier=URL, index=INDEX)
    args.workload_unit = "released frames"
    getattr(external, args.phase)(args)


if __name__ == "__main__":
    main()
