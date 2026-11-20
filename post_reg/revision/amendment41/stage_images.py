"""Stage exact members of the seven prequalified releases without reading annotations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
import zipfile

from lane_b_gate import NAMES, check
from preflight import digest

ROOT = Path.cwd()
RETRY = ROOT / "06_pod_outputs/post_reg/aei_qualification/03_all_retry/attempt_02/verified"
PRIVATE = ROOT / "06_pod_outputs/post_reg/revision/remaining_experiments/private"
ADAPTERS = ROOT / "02_plan/aei_qualification_tasks/.batch-repair/post_reg/supplements/qualification/batch"
SOURCES = {
    "highrpd": "https://data.mendeley.com/public-files/datasets/sywswj7djj/files/518cfb67-17f4-44a9-92a4-6d9d64cf8b83/file_downloaded",
    "uneti": "https://zenodo.org/records/22830366/files/UNETI-RoadDamage.zip?download=1",
    "attain": "https://data.mendeley.com/public-api/zip/nykrzdm74f/download/1",
    "wasit": "https://zenodo.org/records/21167531/files/IRDD_v1.0_final.zip?download=1",
    "m2s": "https://huggingface.co/datasets/DarrylT/M2S-RoAD/resolve/bcc040c0fdff72c3c7f3189849e28b588bb85af1/M2S-RoAD.zip",
}
LOCAL = {"capstone": PRIVATE / "capstone-v11-coco.zip",
         "dashcam": RETRY.parents[1] / "03B/annotation_provenance/attempt_02/.control/dashcam-v14-coco.zip"}


def fetch(name, source_size, expected):
    path = LOCAL.get(name, PRIVATE / f"{name}.zip")
    if not path.exists() and name in LOCAL:
        raise ValueError(f"Missing pinned local {name} source")
    if not path.exists():
        print(f"Fetching pinned {name} release", flush=True)
        partial_path = path.with_suffix(".zip.part")
        req = Request(SOURCES[name], headers={"User-Agent": "curl/8.7.1", "Accept-Encoding": "identity"})
        with urlopen(req, timeout=60) as source, partial_path.open("xb") as target:
            total = 0
            while chunk := source.read(2 * 1024 * 1024):
                total += len(chunk)
                if total > source_size:
                    raise ValueError(f"Source length exceeds the pinned {name} receipt")
                target.write(chunk)
        if total != source_size or digest(partial_path) != expected:
            raise ValueError(f"Source length or SHA-256 differs for {name}")
        partial_path.replace(path)
    if path.stat().st_size != source_size or digest(path) != expected:
        raise ValueError(f"Source length or SHA-256 differs for {name}")
    return zipfile.ZipFile(path)


def range_archive(name, source_size):
    # Import the pinned reader, without invoking its annotation adapters.
    import subprocess
    sealed_batch = "0f48d0ac3fd44714d8bc2aff71d758ac13e4db22"
    git = ["git", "-C", str(ADAPTERS)]
    source = subprocess.check_output(git + ["show", sealed_batch + ":post_reg/supplements/qualification/batch/adapters.py"])
    if source != (ADAPTERS / "adapters.py").read_bytes():
        raise ValueError("The original ranged-member reader differs")
    spec = importlib.util.spec_from_file_location("nine_frozen_range_adapter", ADAPTERS / "adapters.py")
    import sys
    sys.path.insert(0, str(ADAPTERS))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    stream = module.RangeSource(SOURCES[name], source_size)
    archive = zipfile.ZipFile(stream)
    ordered = sorted(archive.infolist(), key=lambda item: item.header_offset)
    ends = {item.filename: (ordered[index + 1].header_offset if index + 1 < len(ordered) else archive.start_dir)
            for index, item in enumerate(ordered)}
    if len(ends) != len(ordered):
        raise ValueError("M2S ZIP contains duplicate archive member names")
    return archive, lambda filename: stream.member(archive.getinfo(filename), ends[filename])


def stage(name, destination):
    scope = RETRY / name
    published = json.loads((scope / "source_evidence.json").read_text())["info"]
    evidence = {item["name"]: item["file_sha256"] for item in
                json.loads((scope / "source_members.json").read_text())}
    files = {part: (scope / "split" / f"{part}.txt").read_text().splitlines()
             for part in ("calibration", "heldout")}
    flat = files["calibration"] + files["heldout"]
    if len(set(flat)) != len(flat) or not set(flat) <= evidence.keys():
        raise ValueError(f"{name} lists differ from the verified member evidence")
    basenames = set()
    name_map = {}
    for index, member in enumerate(flat):
        suffix = PurePosixPath(member).suffix.lower()
        if suffix not in {".jpeg", ".jpg", ".png"}:
            raise ValueError(f"Non-image in {name} split")
        # Stable synthetic filenames avoid basename collisions; the output map restores exact IDs.
        safe = f"{index:06d}{suffix}"
        if safe in basenames:
            raise ValueError("Repeated staged basename")
        basenames.add(safe)
        name_map[member] = safe
    root = destination / name
    (root / "images").mkdir(parents=True, exist_ok=True)
    if name == "m2s":
        if published["declared_object_sha256"] != "3ace64fab8b39bc4363f3d2962fa06c05a87b9fcf4ed4195843a9b16ab180883":
            raise ValueError("M2S ranged-source identity differs")
        archive, read = range_archive(name, published["archive_bytes"])
    else:
        archive = fetch(name, published["archive_bytes"], published["archive_sha256"])
        read = archive.read
    with archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError(f"Repeated archive members for {name}")
        def extract(member):
            target = root / "images" / name_map[member]
            if target.exists():
                if digest(target) != evidence[member]:
                    raise ValueError(f"Staged image checksum differs for {name}")
                return
            content = read(member)
            if hashlib.sha256(content).hexdigest() != evidence[member]:
                raise ValueError(f"Image checksum differs for {name}")
            target.write_bytes(content)
        workers = 12 if name == "m2s" else 1
        with ThreadPoolExecutor(workers) as pool:
            for _ in pool.map(extract, flat):
                pass
    (root / "image_map.json").write_text(json.dumps({value:key for key,value in name_map.items()}, sort_keys=True))
    for part, members in files.items():
        (root / f"{part}.txt").write_text("\n".join(name_map[member] for member in members)+"\n")
    print(f"Verified all {name} image bytes and sealed split membership", flush=True)
    return name, len(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    check()
    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(5) as pool:
        results = dict(pool.map(partial(stage, destination=args.out), sorted(NAMES)))
    (args.out / "STAGED.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
