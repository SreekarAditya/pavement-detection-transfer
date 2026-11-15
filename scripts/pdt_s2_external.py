#!/usr/bin/env python3
"""Pod session 2, step 5 (part 1): external collections, download and preparation. Protocol 3.3.9.

  fetch    download the session-2 files listed in external_inventory.json, verify the host checksums, extract
  prepare  build, per collection, the reference-box table under the frozen class mapping and geometry rule,
           the image pool (one flat folder of links with unique names, used for inference) and a label-only
           overlay sample for a person to check box placement and image orientation.

No detector is run here. Labels may be read (protocol Section 9, step 5).

Frozen rules (written before any external image was processed)
  Pool. Every released image that has a label record (a label file, possibly empty, or an annotation entry).
        Images without any label record are left out and counted.
  Class mapping. Only released classes whose names state longitudinal crack, transverse crack, alligator crack
        or pothole (or the RDD2022 codes D00, D10, D20, D40) are mapped (MAPS below). Boxes of other classes are
        kept in the table with mapped = None; they are not reference boxes.
  Geometry. Image size is the size after applying the EXIF orientation (what Ultralytics and the model see).
        Normalized YOLO boxes are scaled by that size. Polygons (PDD2026) become their enclosing axis-aligned
        box; a polygon needs at least 3 points. Every box is clipped to the image; a box narrower or shorter than
        1 pixel after clipping, or with a non-finite coordinate, is dropped. Exact duplicate boxes (same class and
        coordinates) in one image are dropped, as Ultralytics does for RDD2022. All drops are counted by class.
  Decoding. An image that cannot be fully decoded (for example a truncated file) is left out of the pool and listed;
        a partial decode would show the detector an image that was never captured. (Rule added 15 November 2026 after
        pod session 2 stopped on one truncated EcuPDD file; no detector had been run on any external image.)
  Orientation. Images whose EXIF orientation is not 1 are counted per collection and over-represented in the
        overlay sample, so a person can see whether boxes sit on the damage.
"""
import argparse, hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

SLUGS = {"rome": "Rome version 2", "svrdd": "SVRDD Beijing", "ecupdd": "EcuPDD Ecuador", "ird": "IRD Baghdad",
         "hrp4k": "HRP4K", "kent": "Kent water-filled and dry potholes", "pdd2026": "PDD2026 Indonesia"}
# Excluded before download (inventory label counts, protocol G4): fewer than 150 pothole boxes before any removal.
PRESCREEN_EXCLUDED = {"N-RDD2024, Turkey and Georgia": "129 pothole boxes in the released labels",
                      "UAV-PDD2023 Tianjin": "65 pothole boxes after keeping one of the three augmented copies "
                                             "of each crop (195 with copies)"}
# released class id or name -> RDD2022 code (frozen; checked against the archive's own class list where it has one)
MAPS = {"rome": {0: "D40"},
        "svrdd": {0: "D00", 1: "D10", 2: "D20", 3: "D40"},
        "ecupdd": {0: "D40", 2: "D00", 3: "D10", 5: "D20"},
        "ird": {0: "D00", 1: "D10", 2: "D20", 3: "D40"},
        "hrp4k": {0: "D40"},
        "kent": {"pothole": "D40"},
        "pdd2026": {3: "D40"}}
EXPECTED_NAMES = {"ecupdd": ["Pothole", "Block_Crack", "Longitudinal_Crack", "Transverse_Crack", "Patching",
                             "Alligator_Crack"],
                  "ird": ["D00_LongitudinalCrack", "D10_TransverseCrack", "D20_AlligatorCrack", "D40_Pothole",
                          "SpeedBump"],
                  "pdd2026": ["Depression", "Rutting", "Upheaval", "Pothole", "Patching"]}
IMG = {".jpg", ".jpeg", ".png"}
CLS_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def file_hash(p, algo):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ fetch
def fetch(a):
    inv = json.load(open(a.inventory))
    root = Path(a.root)
    rep = {}
    jobs = []
    for slug in a.only or SLUGS:
        c = next(x for x in inv["collections"] if x["name"] == SLUGS[slug])
        d = root / slug / "download"; d.mkdir(parents=True, exist_ok=True)
        jobs += [(slug, f, d / f["name"]) for f in c["files"] if f["name"] in c["session2_files"]]

    def get(job):
        slug, f, p = job
        if p.exists() and p.stat().st_size == f["size"]:
            return 0
        log(f"{slug}: downloading {f['name']} ({f['size'] / 1e9:.2f} GB)")
        return subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "8", "--retry-delay", "10", "-C", "-",
                               "-o", str(p), f["url"]]).returncode
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:  # parallel transfers; every file is verified below
        codes = list(ex.map(get, jobs))
    for (slug, f, p), code in zip(jobs, codes):
        if code not in (0, 33) and not (p.exists() and p.stat().st_size == f["size"]):
            sys.exit(f"STOP: download failed for {slug}/{f['name']}")
    for slug in a.only or SLUGS:
        c = next(x for x in inv["collections"] if x["name"] == SLUGS[slug])
        d = root / slug / "download"
        rep[slug] = []
        for f in c["files"]:
            if f["name"] not in c["session2_files"]:
                continue
            p = d / f["name"]
            algo, want = f["checksum"].split(":")
            got = file_hash(p, algo)
            ok = got == want and p.stat().st_size == f["size"]
            rep[slug].append(dict(file=f["name"], bytes=p.stat().st_size, algo=algo, expected=want, got=got, ok=ok))
            log(f"{slug}: {f['name']} checksum {'OK' if ok else 'MISMATCH'}")
            if not ok:
                sys.exit(f"STOP: checksum mismatch for {slug}/{f['name']}")
        ex = root / slug / "raw"
        if not (ex / ".extracted").exists():
            ex.mkdir(parents=True, exist_ok=True)
            for f in sorted(d.iterdir()):
                if f.suffix.lower() == ".zip":
                    log(f"{slug}: extracting {f.name}")
                    with zipfile.ZipFile(f) as z:
                        z.extractall(ex)
                else:
                    (ex / f.name).write_bytes(f.read_bytes())
            (ex / ".extracted").write_text("ok")
    json.dump(rep, open(root / "fetch_report.json", "w"), indent=1)


# ------------------------------------------------------------------ parsing helpers
def find_dirs(root, rel_suffix):
    """All directories under root whose path ends with rel_suffix (e.g. 'labels-YOLO')."""
    return sorted(p for p in root.rglob("*") if p.is_dir() and p.as_posix().endswith(rel_suffix)
                  and "__MACOSX" not in p.parts)


def images_in(d):
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG and not p.name.startswith("._"))


def img_info(p):
    """(width, height, EXIF orientation, fully decodable) with width and height after the orientation."""
    from PIL import Image
    try:
        with Image.open(p) as im:
            w, h = im.size
            try:
                o = int(im.getexif().get(0x0112, 1) or 1)
            except Exception:
                o = 1
            im.load()
        ok = True
    except Exception:
        return 0, 0, 1, False
    if o in (5, 6, 7, 8):
        w, h = h, w
    return w, h, o, ok


def read_yolo(txt, poly=False):
    """Returns list of (cls, x1, y1, x2, y2) normalized, and number of malformed rows."""
    rows, bad = [], 0
    if txt is None or not txt.exists():
        return rows, bad
    for line in txt.read_text(errors="replace").splitlines():
        v = line.split()
        if not v:
            continue
        try:
            c = int(float(v[0])); xs = [float(t) for t in v[1:]]
        except ValueError:
            bad += 1; continue
        if poly:
            if len(xs) < 6 or len(xs) % 2:
                bad += 1; continue
            px, py = xs[0::2], xs[1::2]
            rows.append((c, min(px), min(py), max(px), max(py)))
        else:
            if len(xs) != 4:
                bad += 1; continue
            x, y, w, h = xs
            rows.append((c, x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return rows, bad


def yolo_pairs(img_dir, lab_dir):
    """(image, label file or None) for every image in img_dir."""
    out = []
    for im in images_in(img_dir):
        t = lab_dir / (im.stem + ".txt")
        out.append((im, t if t.exists() else None))
    return out


def collect(slug, raw):
    """Returns a list of records: (image path, list of (raw_class, x1, y1, x2, y2, normalized?)), malformed count,
    names found in the archive (for the class-list check)."""
    recs, bad, names = [], 0, None
    if slug in ("rome", "svrdd", "ecupdd", "hrp4k", "pdd2026"):
        if slug == "rome":
            ids = find_dirs(raw, "data/images"); lds = find_dirs(raw, "data/labels-YOLO")
            assert len(ids) == 1 and len(lds) == 1, (ids, lds)
            pairs = yolo_pairs(ids[0], lds[0])
        elif slug == "svrdd":
            pairs = []
            for idir in find_dirs(raw, "/images"):
                if "SVRDD_YOLO" in idir.parts:
                    pairs += yolo_pairs(idir, idir.parent / "labels")
        elif slug == "ecupdd":
            pairs = []
            for sp in ("train", "valid"):
                ids = find_dirs(raw, f"EPDD/{sp}/images"); assert len(ids) == 1, ids
                pairs += yolo_pairs(ids[0], ids[0].parent / "labels")
            ys = list(raw.rglob("EPDD/data.yaml"))
            if ys:
                import yaml
                n = yaml.safe_load(open(ys[0]))["names"]; names = list(n.values()) if isinstance(n, dict) else list(n)
        elif slug == "hrp4k":
            pairs = []
            for sp in ("train", "valid", "test"):
                ids = find_dirs(raw, f"HRP4K/{sp}/images"); assert len(ids) == 1, ids
                pairs += yolo_pairs(ids[0], ids[0].parent / "labels")
        else:  # pdd2026
            pairs = []
            for sp in ("train", "val", "test"):
                ids = find_dirs(raw, f"FinalDataset_1024/images/{sp}"); assert len(ids) == 1, ids
                pairs += yolo_pairs(ids[0], ids[0].parent.parent / "labels" / sp)
            ys = list(raw.rglob("FinalDataset_1024/data.yaml"))
            if ys:
                import yaml
                n = yaml.safe_load(open(ys[0]))["names"]; names = list(n.values()) if isinstance(n, dict) else list(n)
        for im, t in pairs:
            if t is None:
                recs.append((im, None)); continue
            rows, b = read_yolo(t, poly=(slug == "pdd2026")); bad += b
            recs.append((im, [(c, x1, y1, x2, y2, True) for c, x1, y1, x2, y2 in rows]))
    elif slug == "ird":
        ids = [d for d in find_dirs(raw, "/images") if not any(x.startswith("labels") for x in d.parts)]
        lds = find_dirs(raw, "labels_bbox")
        assert len(lds) == 1, lds
        imgs = sorted({p for d in ids for p in images_in(d)}, key=lambda p: p.name)
        assert len({p.name for p in imgs}) == len(imgs), "IRD image names not unique after merging the archives"
        for im in imgs:
            t = lds[0] / (im.stem + ".txt")
            if not t.exists():
                recs.append((im, None)); continue
            rows, b = read_yolo(t); bad += b
            recs.append((im, [(c, x1, y1, x2, y2, True) for c, x1, y1, x2, y2 in rows]))
        ys = sorted(raw.rglob("data.yaml"))
        if ys:
            import yaml
            n = yaml.safe_load(open(ys[0]))["names"]; names = list(n.values()) if isinstance(n, dict) else list(n)
    elif slug == "kent":
        ids = find_dirs(raw, "/IMG"); xds = find_dirs(raw, "/XML")
        assert len(ids) == 1 and len(xds) == 1, (ids, xds)
        for im in images_in(ids[0]):
            x = xds[0] / (im.stem + ".xml")
            if not x.exists():
                recs.append((im, None)); continue
            rows = []
            for o in ET.parse(x).getroot().iter("object"):
                bb = o.find("bndbox")
                try:
                    v = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
                except Exception:
                    bad += 1; continue
                rows.append((o.find("name").text.strip(), *v, False))
            recs.append((im, rows))
    else:
        raise ValueError(slug)
    return recs, bad, names


# ------------------------------------------------------------------ prepare
def prepare(a):
    root = Path(a.root)
    summary = {}
    for slug in a.only or SLUGS:
        raw = root / slug / "raw"
        out = root / slug / "pool"; (out / "images").mkdir(parents=True, exist_ok=True)
        recs, bad, names = collect(slug, raw)
        if slug in EXPECTED_NAMES and names is not None and names != EXPECTED_NAMES[slug]:
            sys.exit(f"STOP: {slug} class list {names} differs from the inventory {EXPECTED_NAMES[slug]}")
        mp = MAPS[slug]
        rows, imgs = [], []
        drops = dict(geometry={}, duplicate={})
        no_label = 0
        from multiprocessing import Pool
        with Pool(min(32, os.cpu_count() or 8)) as pool:
            infos = pool.map(img_info, [im for im, b in recs], chunksize=16)
        undecodable = []
        for (im, boxes), (w, h, orient, ok) in zip(recs, infos):
            if boxes is None:
                no_label += 1; continue
            rel = im.relative_to(raw).as_posix()
            flat = f"{slug}__" + rel.replace("/", "__").replace(" ", "_")
            if not ok:
                undecodable.append(rel); continue
            imgs.append(dict(file=flat, source=rel, width=w, height=h, exif_orientation=orient, n_boxes=len(boxes)))
            seen = set()
            for c, x1, y1, x2, y2, norm in boxes:
                m = mp.get(c.strip().lower() if isinstance(c, str) else c)
                if norm:
                    x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
                key = m or f"unmapped:{c}"
                if not all(np.isfinite([x1, y1, x2, y2])):
                    drops["geometry"][key] = drops["geometry"].get(key, 0) + 1; continue
                x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
                y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    drops["geometry"][key] = drops["geometry"].get(key, 0) + 1; continue
                k = (c, round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))
                if k in seen:
                    drops["duplicate"][key] = drops["duplicate"].get(key, 0) + 1; continue
                seen.add(k)
                rows.append(dict(file=flat, raw_class=str(c), mapped=m, cls=CLS_ID.get(m, -1),
                                 x1=x1, y1=y1, x2=x2, y2=y2))
            link = out / "images" / flat
            if not link.exists():
                os.symlink(im.resolve(), link)
        I = pd.DataFrame(imgs); L = pd.DataFrame(rows, columns=["file", "raw_class", "mapped", "cls", "x1", "y1",
                                                                "x2", "y2"])
        I.to_csv(out / "images.csv", index=False); L.to_parquet(out / "labels.parquet")
        cnt = {k: int((L.mapped == k).sum()) for k in CLS_ID}
        s = dict(images=len(I), images_without_label_record=no_label, undecodable_images=undecodable,
                 malformed_label_rows=bad,
                 exif_orientation_not_1=int((I.exif_orientation != 1).sum()) if len(I) else 0,
                 boxes_by_class=cnt, unmapped_boxes=int(L.mapped.isna().sum()),
                 unmapped_by_raw_class=L[L.mapped.isna()].raw_class.value_counts().to_dict(),
                 drops=drops, class_list_in_archive=names, mapping={str(k): v for k, v in mp.items()},
                 labels_sha256=file_hash(out / "labels.parquet", "sha256"),
                 images_csv_sha256=file_hash(out / "images.csv", "sha256"))
        summary[slug] = s
        log(f"{slug}: {len(I)} images, boxes {cnt}, unmapped {s['unmapped_boxes']}, "
            f"EXIF orientation not 1: {s['exif_orientation_not_1']}, no label record: {no_label}, "
            f"not decodable: {len(undecodable)}")
        overlays(slug, raw, I, L, root / "overlays" / slug)
    json.dump(dict(summary=summary, prescreen_excluded=PRESCREEN_EXCLUDED), open(root / "prepare_report.json", "w"),
              indent=1)


def overlays(slug, raw, I, L, outdir, n=12):
    """Label-only overlay sample (seed 2026): up to 6 images with EXIF orientation not 1, the rest at random."""
    from PIL import Image, ImageDraw, ImageOps
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    has = I[I.n_boxes > 0]
    rot = has[has.exif_orientation != 1]
    pick = list(rot.file.iloc[rng.permutation(len(rot))[:6]]) if len(rot) else []
    rest = has[~has.file.isin(pick)]
    pick += list(rest.file.iloc[rng.permutation(len(rest))[:n - len(pick)]])
    col = {"D00": "red", "D10": "blue", "D20": "orange", "D40": "lime"}
    for f in pick:
        src = raw / I.set_index("file").loc[f, "source"]
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        d = ImageDraw.Draw(im); lw = max(2, im.width // 400)
        for r in L[L.file == f].itertuples():
            m = r.mapped if isinstance(r.mapped, str) else None
            d.rectangle([r.x1, r.y1, r.x2, r.y2], outline=col.get(m, "white"), width=lw)
            d.text((r.x1 + 3, r.y1 + 3), m or f"x{r.raw_class}", fill=col.get(m, "white"))
        im.thumbnail((1600, 1600)); im.save(outdir / (f + ".jpg"), quality=88)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "prepare"])
    ap.add_argument("--inventory", default="/workspace/run/inputs/external_inventory.json")
    ap.add_argument("--root", default="/workspace/run/ext")
    ap.add_argument("--only", nargs="*", choices=list(SLUGS))
    a = ap.parse_args()
    fetch(a) if a.cmd == "fetch" else prepare(a)
