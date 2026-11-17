#!/usr/bin/env python3
"""Experiment 18: the procedure on road networks that were not used anywhere in the study (task 09).

  download each candidate's files, checked against the SHA-256 values of the rule document, then extracted
  prepare  each candidate release (rules/exp18_candidates.csv) read with the frozen class mapping of the rule
           document (a generic "crack" stays unmapped); YOLO text or Pascal VOC labels; same geometry and
           duplicate checks as the registered preparation (scripts/pdt_s2_external.py, whose readers are reused)
  overlap  overlap removal and blocks with the registered rule (scripts/pdt_s2_g4.py: pHash 4 bits or fewer or
           SSCD cosine 0.75 or more; blocks at cosine 0.65), against gate G4's RDD2022 reference set (C0 for
           training and validation, the locked G4 test descriptors) AND the seven collections already used; track eligibility (four-class: 150 boxes of each class; merged crack: 450
           crack boxes; pothole: 150 boxes; each with at least 20 blocks); block split into calibration and
           held-out parts; held-out labels written to private/heldout_labels_<network>.parquet, hashed, and
           not read again before phase D
  decide   Phase C: threshold, pilot estimate and accept-or-adapt decision from calibration labels only
  adapt    Phase C, only for networks where the procedure said adapt: fine-tune and predict (graphics card)
  commit   Phase C: exp18_commitments.json (thresholds, stated recalls with intervals, decisions, adaptation)
  score    Phase D, after the commitments: held-out alert volume and true recall for every committed row, the
           two floors and the ceiling, with experiment 1's code; the procedure is recomputed and must equal the
           committed rows
"""
import argparse, json, re, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402

CLS_ID = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
TRACK_CLASS_SET = {"four_class": "four_class", "merged_crack": "merged_crack", "pothole": "pothole"}


def norm_name(c):
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", str(c).strip().lower()))


def frozen_map(params):
    m = {norm_name(k): v for k, v in params["exp18"]["mapping"].items()}
    bad = {v for v in m.values() if v not in CLS_ID}
    if bad:
        sys.exit(f"STOP: mapping targets {bad} are not RDD2022 classes")
    return m


# ------------------------------------------------------------------ download and prepare
REQUIRED = ["slug", "release", "team", "format", "images_dir", "labels_dir", "license", "url", "download_urls", "sha256"]


def check_candidates(C):
    """Every required field filled (no '?' or blank) before anything is downloaded or read."""
    miss = [c for c in REQUIRED if c not in C.columns]
    if miss:
        sys.exit(f"STOP: candidates file lacks columns {miss}")
    bad = [(r.slug, c) for r in C.itertuples() for c in REQUIRED
           if pd.isna(getattr(r, c)) or str(getattr(r, c)).strip() in ("", "?")]
    if bad:
        sys.exit(f"STOP: candidates file has unfilled fields {bad}")
    for col in ("parent_regex", "copy_regex"):
        if col not in C.columns:
            continue
        for r in C.itertuples():
            v = getattr(r, col)
            if pd.isna(v) or str(v).strip() == "":
                continue
            if str(v).strip() == "?":
                sys.exit(f"STOP: candidates file has unfilled field ({r.slug}, {col})")
            try:
                rx = re.compile(str(v))
            except re.error as e:
                sys.exit(f"STOP: {r.slug} {col} does not compile: {e}")
            if rx.groups < 1:
                sys.exit(f"STOP: {r.slug} {col} needs one capture group (the source or parent identifier)")


def download(a, params):
    """Every candidate's files (candidates CSV: download_urls and sha256, ';'-separated, same order) into
    <raw>/<slug>/download, checked by SHA-256, then extracted into <raw>/<slug>/files. Re-running only checks."""
    import subprocess, tarfile, zipfile
    C = pd.read_csv(a.candidates); rep = {}
    check_candidates(C)
    for row in C.itertuples():
        urls, shas = str(row.download_urls).split(";"), str(row.sha256).split(";")
        if len(urls) != len(shas):
            sys.exit(f"STOP: {row.slug}: {len(urls)} URLs but {len(shas)} hashes")
        d = Path(a.raw) / row.slug / "download"; d.mkdir(parents=True, exist_ok=True)
        x = Path(a.raw) / row.slug / "files"
        rep[row.slug] = []
        for u, h in zip(urls, shas):
            f = d / (Path(u.split("?")[0]).name or "file")
            def digest(f_, h_):   # "md5:<hex>" where the publisher gives only MD5 (Zenodo); SHA-256 otherwise
                if h_.startswith("md5:"):
                    import hashlib
                    m_ = hashlib.md5()
                    with open(f_, "rb") as fh:
                        for blk in iter(lambda: fh.read(1 << 20), b""):
                            m_.update(blk)
                    return "md5:" + m_.hexdigest()
                return PC.sha(f_)
            if not (f.exists() and digest(f, h) == h):
                r = subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "8", "--retry-delay", "10", "-o", str(f), u])
                if r.returncode:
                    sys.exit(f"STOP: download failed: {row.slug} {u}")
            got = digest(f, h)
            if got != h:
                sys.exit(f"STOP: SHA-256 differs for {row.slug} {f.name}")
            rep[row.slug].append(dict(url=u, file=f.name, bytes=f.stat().st_size, checksum=got,
                                      sha256=PC.sha(f)))
        if not (x / ".extracted").exists():
            x.mkdir(parents=True, exist_ok=True)
            for f in sorted(d.iterdir()):
                if zipfile.is_zipfile(f):
                    with zipfile.ZipFile(f) as z:
                        z.extractall(x)
                elif tarfile.is_tarfile(f):
                    with tarfile.open(f) as t:
                        t.extractall(x, filter="data")
                else:
                    shutil.copy(f, x / f.name)
            (x / ".extracted").write_text("ok")
        PC.log(f"{row.slug}: {len(rep[row.slug])} files checked")
    PC.jdump(rep, Path(a.ext18) / "exp18_download.json")


def cand_path(a, row, p):
    """Candidate folders are given relative to <raw>/<slug>/files (absolute paths are used as they are)."""
    p = Path(p)
    return p if p.is_absolute() else Path(a.raw) / row.slug / "files" / p


def parts(a, row):
    """(images folder, labels location) pairs of a candidate: ';'-separated in images_dir and labels_dir, same
    count (for example the train and valid folders of one release). For COCO, the label location is the JSON file."""
    if str(row.images_dir).strip() == "auto" or str(row.labels_dir).strip() == "auto":
        return coco_auto_parts(a, row)
    im = [x.strip() for x in str(row.images_dir).split(";")]
    lb = [x.strip() for x in str(row.labels_dir).split(";")]
    if len(im) != len(lb):
        sys.exit(f"STOP: {row.slug}: {len(im)} image folders but {len(lb)} label locations")
    return [(cand_path(a, row, i), cand_path(a, row, l)) for i, l in zip(im, lb)]


def coco_auto_parts(a, row):
    """Layout rule for COCO releases whose folder names are not known before download ("auto" in both
    images_dir and labels_dir): every JSON file under <raw>/<slug>/files that holds "images", "annotations" and
    "categories" is one part, in sorted path order; its images are looked up in the JSON file's own folder. Only
    file names and the JSON keys are read here."""
    if row.format != "coco" or str(row.images_dir).strip() != "auto" or str(row.labels_dir).strip() != "auto":
        sys.exit(f"STOP: {row.slug}: 'auto' layout is defined for COCO only, in both images_dir and labels_dir")
    root = Path(a.raw) / row.slug / "files"
    out = []
    for j in sorted(root.rglob("*.json")):
        if any(q.startswith(".") or q == "__MACOSX" for q in j.relative_to(root).parts):
            continue
        try:
            J = json.load(open(j))
        except Exception:
            continue
        if isinstance(J, dict) and all(k in J for k in ("images", "annotations", "categories")):
            out.append((j.parent, j))
    if not out:
        sys.exit(f"STOP: {row.slug}: no COCO annotation file found under {root}")
    return out


def unlisted_images(P, X):
    """Image files in the part folders that no COCO file lists (reported, not used)."""
    listed = set()
    for img_dir, lab in P:
        for im in json.load(open(lab)).get("images", []):
            listed.add((img_dir / im["file_name"]).resolve())
    return sorted({str(f) for d, _ in P for f in X.images_in(d) if f.resolve() not in listed})


def read_candidate(a, row, X):
    """Records (image path, source name, [(raw name, x1, y1, x2, y2, normalized)] or None) and the count of
    malformed label rows. The source name is the image path relative to its part folder, prefixed by the part
    number when there is more than one part."""
    recs, bad = [], 0
    P = parts(a, row)
    for k, (img_dir, lab) in enumerate(P):
        pre = f"part{k}/" if len(P) > 1 else ""
        if row.format == "yolo":
            names = str(row.class_names).split(";")
            for im, txt in X.yolo_pairs(img_dir, lab):
                src = pre + im.relative_to(img_dir).as_posix()
                if txt is None:
                    recs.append((im, src, None)); continue
                rows, b = X.read_yolo(txt); bad += b
                out = []
                for c, *v in rows:
                    if 0 <= c < len(names):
                        out.append((names[c], *v, True))
                    else:
                        bad += 1
                recs.append((im, src, out))
        elif row.format == "voc":
            import xml.etree.ElementTree as ET
            for im in X.images_in(img_dir):
                src = pre + im.relative_to(img_dir).as_posix()
                x = lab / (im.stem + ".xml")
                if not x.exists():
                    recs.append((im, src, None)); continue
                out = []
                for o in ET.parse(x).getroot().iter("object"):
                    bb = o.find("bndbox")
                    try:
                        v = [float(bb.find(q).text) for q in ("xmin", "ymin", "xmax", "ymax")]
                    except Exception:
                        bad += 1; continue
                    out.append((o.find("name").text.strip(), *v, False))
                recs.append((im, src, out))
        elif row.format == "coco":
            J = json.load(open(lab))
            cats = {c["id"]: c["name"] for c in J.get("categories", [])}
            by = {}
            for an in J.get("annotations", []):
                try:
                    x, y, w, h = (float(t) for t in an["bbox"])
                    by.setdefault(an["image_id"], []).append((cats[an["category_id"]], x, y, x + w, y + h, False))
                except Exception:
                    bad += 1
            for im in J.get("images", []):
                f = img_dir / im["file_name"]
                if not f.exists():
                    bad += 1; continue
                # an image listed in the COCO file is labeled; no annotation means no box on it
                recs.append((f, pre + Path(im["file_name"]).as_posix(), by.get(im["id"], [])))
        else:
            sys.exit(f"STOP: format {row.format} of {row.slug} is not yolo, voc or coco")
    return recs, bad


def prepare(a, params):
    X = __import__("pdt_s2_external")
    mp = frozen_map(params)
    C = pd.read_csv(a.candidates)
    check_candidates(C)
    clash = set(C.slug) & set(PC.EXT_ALL)
    if clash:
        sys.exit(f"STOP: candidate names {clash} reuse a collection name")
    summ = {}
    for row in C.itertuples():
        out = Path(a.ext18) / row.slug / "pool"; (out / "images").mkdir(parents=True, exist_ok=True)
        recs, bad = read_candidate(a, row, X)
        infos = PC.pmap(X.img_info, [im for im, _, _ in recs])
        imgs, rows, drops, no_label, undecodable = [], [], dict(geometry=0, duplicate=0), 0, []
        for (im, rel, boxes), (w, h, orient, ok) in zip(recs, infos):
            if boxes is None:
                no_label += 1; continue
            flat = f"{row.slug}__" + rel.replace("/", "__").replace(" ", "_")
            if not ok:
                undecodable.append(rel); continue
            imgs.append(dict(file=flat, source=rel, width=w, height=h, exif_orientation=orient, n_boxes=len(boxes)))
            seen = set()
            for c, x1, y1, x2, y2, nrm in boxes:
                m = mp.get(norm_name(c))
                if nrm:
                    x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
                if not all(np.isfinite([x1, y1, x2, y2])):
                    drops["geometry"] += 1; continue
                x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
                y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    drops["geometry"] += 1; continue
                key = (c, round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))
                if key in seen:
                    drops["duplicate"] += 1; continue
                seen.add(key)
                rows.append(dict(file=flat, raw_class=str(c), mapped=m, cls=CLS_ID.get(m, -1),
                                 x1=x1, y1=y1, x2=x2, y2=y2))
            link = out / "images" / flat
            if not link.exists():
                link.symlink_to(Path(im).resolve())
        I = pd.DataFrame(imgs)
        L = pd.DataFrame(rows, columns=["file", "raw_class", "mapped", "cls", "x1", "y1", "x2", "y2"])
        if I.file.duplicated().any():
            sys.exit(f"STOP: duplicate flattened names in {row.slug}")
        I.to_csv(out / "images.csv", index=False); L.to_parquet(out / "labels.parquet")
        summ[row.slug] = dict(release=row.release, team=row.team, license=row.license, url=row.url,
                              images=len(I), images_without_label_record=no_label, undecodable=undecodable,
                              malformed_label_rows=bad, dropped=drops,
                              boxes_by_class={k: int((L.mapped == k).sum()) for k in CLS_ID},
                              unmapped_by_raw_class=L[L.mapped.isna()].raw_class.value_counts().to_dict(),
                              labels_sha256=PC.sha(out / "labels.parquet"))
        if row.format == "coco":
            P = parts(a, row); root = Path(a.raw) / row.slug / "files"
            summ[row.slug]["parts"] = [dict(images_dir=str(d.relative_to(root)) if d.is_relative_to(root) else str(d),
                                            labels=str(j.relative_to(root)) if j.is_relative_to(root) else str(j))
                                       for d, j in P]
            summ[row.slug]["images_not_in_any_coco_file"] = len(unlisted_images(P, X))
        PC.log(f"{row.slug}: {len(I)} images, boxes {summ[row.slug]['boxes_by_class']}")
    PC.jdump(summ, Path(a.ext18) / "exp18_prepare.json")


# ------------------------------------------------------------------ overlap, blocks, eligibility, split, seal
def rdd_reference(a, c0, have_desc):
    """The registered RDD2022 reference set of gate G4: training and validation images from C0 (pod session 1:
    index.csv, desc_train.npy, desc_val.npy) and test images from the locked G4 outputs (g4_index.csv rows
    'rdd2022_test', desc_test.npy, same order)."""
    c0d = Path(a.c0)
    idx = pd.read_csv(c0d / "index.csv")
    g = pd.read_csv(Path(a.root) / "s2" / "g4" / "g4_index.csv"); t = g[g.collection == "rdd2022_test"]
    names = [f"rdd2022_{s_}:{f}" for s_, f in zip(idx.split, idx.file)] + [f"rdd2022_test:{f}" for f in t.file]
    ph = pd.concat([idx.phash, t.phash], ignore_index=True)
    d = None
    if have_desc:
        d = np.concatenate([np.load(c0d / "desc_train.npy"), np.load(c0d / "desc_val.npy"),
                            np.load(Path(a.root) / "s2" / "g4" / "desc_test.npy")]).astype(np.float32)
        if len(d) != len(ph):
            sys.exit("STOP: RDD2022 reference descriptors and hashes differ in length")
    files = {f: PC.sha(c0d / f) for f in ("index.csv", "desc_train.npy", "desc_val.npy", "c0_meta.json")
             if (c0d / f).exists()}
    return ph, d, names, files


def used_reference(root, have_desc):
    """The seven collections already used: hashes from g4_index.csv and descriptors desc_<slug>.npy (the rows of
    each slug in g4_index.csv are in the same order as its descriptor file)."""
    g = pd.read_csv(Path(root) / "s2" / "g4" / "g4_index.csv")
    ph, ds, names = [], [], []
    for c in PC.EXT_ALL:
        h = g[g.collection == c]
        ph.append(h.phash); names += [f"{c}:{f}" for f in h.file]
        if have_desc:
            d = np.load(Path(root) / "s2" / "g4" / f"desc_{c}.npy").astype(np.float32)
            if len(d) != len(h):
                sys.exit(f"STOP: descriptor rows of {c} do not match g4_index.csv")
            ds.append(d)
    return pd.concat(ph, ignore_index=True), (np.concatenate(ds) if have_desc else None), names


def dihedral_phashes(path):
    """Perceptual hashes of the image under the seven non-identity rotations and reflections (hex strings)."""
    from PIL import Image, ImageOps
    import imagehash
    T = Image.Transpose
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return [str(imagehash.phash(im.transpose(t))) for t in
                (T.ROTATE_90, T.ROTATE_180, T.ROTATE_270, T.FLIP_LEFT_RIGHT, T.FLIP_TOP_BOTTOM, T.TRANSPOSE, T.TRANSVERSE)]


def transformed_copies(paths, names, ph, c0, thr, sources=None, copy_regex=""):
    """Groups of released copies of one image: rotated or reflected copies (perceptual-hash distance at most thr
    between one image under a rotation or reflection and the other as released), and, where the release names its
    copies after the source image (copy_regex, group 1 = source image identifier, for example an export tool's
    augmented copies), all images with the same identifier. Returns the names to drop (every member of a group
    except the one with the smallest name) and the number of groups."""
    T = np.stack([c0.hex_to_u64(col) for col in zip(*PC.pmap(dihedral_phashes, paths))], 1)   # n x 7
    n = len(names); parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for t in range(T.shape[1]):
        for i in range(0, n, 512):
            d = c0.popcount64(T[i:i + 512, t][:, None] ^ ph[None, :])
            qi, rj = np.nonzero(d <= thr)
            for x, y in zip(qi + i, rj):
                if x != y:
                    parent[find(x)] = find(y)
    if sources is not None and isinstance(copy_regex, str) and copy_regex.strip():
        import re as _re
        rx = _re.compile(copy_regex); first = {}
        for i, s_ in enumerate(sources):
            m = rx.search(s_)
            if m:
                j = first.setdefault(m.group(1), i)
                if j != i:
                    parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(names[i])
    return sorted(m for g in groups.values() if len(g) > 1 for m in sorted(g)[1:]), \
        sum(1 for g in groups.values() if len(g) > 1)


def parent_links(sources, regex):
    """Pairs of image positions that share a parent identifier taken from the source name by regex (group 1)."""
    if not isinstance(regex, str) or not regex.strip():
        return set()
    import re as _re
    rx = _re.compile(regex); by = {}
    for i, s_ in enumerate(sources):
        m = rx.search(s_)
        if m:
            by.setdefault(m.group(1), []).append(i)
    return {(g[j], g[j + 1]) for g in by.values() for j in range(len(g) - 1)}


def overlap(a, params):
    PC.locked_scripts(a.root)
    G = __import__("pdt_s2_g4"); c0 = sys.modules["pdt_c0"]
    import torch
    a.dev = f"cuda:{a.device}" if a.device != "cpu" and torch.cuda.is_available() else "cpu"
    a.model, mfile = (None, None) if a.hash_only_test else c0.load_sscd(a.dev)
    dec = json.load(open(Path(a.root) / "s2" / "g4" / "g4_g6_decision.json"))
    if a.model is None and dec.get("descriptor") is not None and not a.hash_only_test:
        sys.exit("STOP: SSCD unavailable but the registered overlap rule used descriptors")
    meta = json.load(open(Path(a.c0) / "c0_meta.json"))
    if a.model is not None:
        if PC.sha(mfile) != meta["descriptor_file_sha256"]:
            sys.exit("STOP: SSCD file differs from the one used in pod session 1")
    rph, rdesc, rnames, c0_files = rdd_reference(a, c0, a.model is not None)
    if len(rph) != dec.get("reference_images"):
        msg = f"RDD2022 reference images {len(rph)}, registered reference set {dec.get('reference_images')}"
        if not a.hash_only_test:
            sys.exit("STOP: " + msg)
        PC.log("test mode: " + msg)
    uph, udesc, unames = used_reference(a.root, rdesc is not None)
    ref_ph = np.concatenate([c0.hex_to_u64(rph), c0.hex_to_u64(uph)])
    ref_d = np.concatenate([rdesc, udesc]) if rdesc is not None else None
    ref_names = rnames + unames
    e18 = params["exp18"]; tr = e18["tracks"]; minb = e18["min_blocks"]
    C = pd.read_csv(a.candidates)
    rep = dict(reference=dict(rdd2022_images=len(rph), used_collection_images=len(unames),
                              descriptor=None if a.model is None else "sscd_disc_mixup", c0_files_sha256=c0_files,
                              rule=dict(link=f"pHash <= {G.LINK_PH} or SSCD cosine >= {G.LINK_COS}",
                                        blocks=f"pHash <= {G.LINK_PH} or SSCD cosine >= {G.BLOCK_COS}")),
               networks={})
    split_dir = Path(a.ext18) / "rules_split" / "split"; split_dir.mkdir(parents=True, exist_ok=True)
    all_blocks, lists = [], Path(a.ext18) / "lists"; lists.mkdir(exist_ok=True)
    for row in C.itertuples():
        slug = row.slug; pool = Path(a.ext18) / slug / "pool"
        I = pd.read_csv(pool / "images.csv"); L = pd.read_parquet(pool / "labels.parquet")
        paths = [str(pool / "images" / f) for f in I.file]
        h = G.hash_table(paths, I.file.tolist())
        if h.phash.isna().any():
            sys.exit(f"STOP: {int(h.phash.isna().sum())} images of {slug} could not be read")
        qph = c0.hex_to_u64(h.phash); n_released = len(I)
        # released rotated or reflected copies of the same image are collapsed to one before anything is counted
        tdrop, tgroups = transformed_copies(paths, I.file.tolist(), qph, c0, G.LINK_PH,
                                            I.source.astype(str).tolist(), getattr(row, "copy_regex", ""))
        if tdrop:
            k_ = ~I.file.isin(tdrop).to_numpy()
            I = I[k_].reset_index(drop=True); L = L[L.file.isin(set(I.file))].reset_index(drop=True)
            paths = [p_ for p_, k1 in zip(paths, k_) if k1]; qph = qph[k_]
        # a release known to ship transformed copies states the largest share of images the collapse may keep;
        # a collapse that keeps more is reported as failed and the network is dropped (the rule is not changed)
        mks = getattr(row, "max_kept_share", "")
        collapse_failed = (str(mks).strip() not in ("", "nan")
                           and len(I) > float(mks) * n_released)
        qd = c0.embed(paths, a.model, a.dev, workers=G.NW) if a.model is not None else None
        lk = G.links(qd, ref_d, qph, ref_ph, G.LINK_COS, G.LINK_PH)
        rm = sorted({I.file.iloc[q] for q, _ in lk})
        linked_to = pd.Series([ref_names[r].split(":")[0] for _, r in lk]).value_counts().to_dict() if lk else {}
        keep = ~I.file.isin(rm)
        I2 = I[keep].reset_index(drop=True); L2 = L[L.file.isin(set(I2.file))].reset_index(drop=True)
        qd2 = qd[keep.to_numpy()] if qd is not None else None; qph2 = qph[keep.to_numpy()]
        wl = G.links(qd2, qd2, qph2, qph2, G.BLOCK_COS, G.LINK_PH, same=True)
        pl = parent_links(I2.source.astype(str).tolist(), getattr(row, "parent_regex", ""))
        comp = G.components(len(I2), set(wl) | pl)
        blocks = pd.DataFrame(dict(collection=slug, file=I2.file, block=[f"{slug}_{c}" for c in comp]))
        all_blocks.append(blocks)
        cnt = {k: int((L2.mapped == k).sum()) for k in CLS_ID}
        nb = int(blocks.block.nunique())
        elig = dict(four_class=all(cnt[k] >= tr["four_class"]["min_per_class"] for k in CLS_ID) and nb >= minb,
                    merged_crack=(cnt["D00"] + cnt["D10"] + cnt["D20"]) >= tr["merged_crack"]["min_total"] and nb >= minb,
                    pothole=cnt["D40"] >= tr["pothole"]["min"] and nb >= minb)
        if collapse_failed:
            elig = {t: False for t in elig}
        cal, held = PC.make_split(I2.file.tolist(), dict(zip(blocks.file, blocks.block)),
                                  params["split"]["calibration_share"], params["split"]["seed"],
                                  params["split"].get("balance", "images"))
        (split_dir / f"{slug}_calibration.txt").write_text("\n".join(cal) + "\n")
        (split_dir / f"{slug}_heldout.txt").write_text("\n".join(held) + "\n")
        (pool.parent / "pool_final.txt").write_text("\n".join(I2.file) + "\n")
        (lists / f"{slug}.txt").write_text("\n".join(str(pool / "images" / f) for f in I2.file) + "\n")
        # seal: held-out labels leave the open folder; only their hash is recorded
        open_d = pool.parent / "open"; open_d.mkdir(exist_ok=True)
        hp = heldout_path(a.ext18, slug); hp.parent.mkdir(exist_ok=True)
        L2[L2.file.isin(set(cal))].to_parquet(open_d / "labels_calibration.parquet")
        L2[L2.file.isin(set(held))].to_parquet(hp)
        (pool / "labels.parquet").unlink()  # the full label file would expose the held-out labels
        rep["networks"][slug] = dict(release=row.release, team=row.team, transformed_copies_dropped=len(tdrop),
                                     transformed_copy_groups=tgroups, parent_links=len(pl),
                                     images_released=n_released, images_before=len(I),
                                     collapse_failed=bool(collapse_failed),
                                     removed_linked=len(rm), removed_linked_to=linked_to, images_after=len(I2),
                                     boxes_after=cnt, blocks=nb,
                                     largest_block_share=round(int(blocks.block.value_counts().max()) / max(len(I2), 1), 4)
                                     if len(I2) else 0,
                                     eligible=elig, calibration_images=len(cal), heldout_images=len(held),
                                     heldout_labels_sha256=PC.sha(hp),
                                     calibration_labels_sha256=PC.sha(open_d / "labels_calibration.parquet"))
        PC.log(f"{slug}: removed {len(rm)} {linked_to}; {len(I2)} images, {nb} blocks, boxes {cnt}; eligible {elig}")
    pd.concat(all_blocks).to_csv(Path(a.ext18) / "blocks.csv", index=False)
    teams = {t: sorted({rep["networks"][s]["team"] for s in rep["networks"] if rep["networks"][s]["eligible"][t]})
             for t in tr}
    rep["eligible_by_track"] = {t: sorted(s for s in rep["networks"] if rep["networks"][s]["eligible"][t]) for t in tr}
    rep["distinct_teams_by_track"] = teams
    PC.jdump(rep, Path(a.ext18) / "exp18_eligibility.json")


# ------------------------------------------------------------------ phases C and D
def heldout_path(ext18, slug):
    return Path(ext18) / "private" / f"heldout_labels_{slug}.parquet"


class Root18(PC.Root):
    """Locked root for internal data (floors); the new networks from the experiment 18 folder. sealed=True: only
    calibration labels are readable (phase C)."""

    def __init__(self, root, ext18, pred, sealed=False):
        super().__init__(root)
        self.e, self.p, self.sealed = Path(ext18), Path(pred), sealed

    def pool(self, c):
        return sorted((self.e / c / "pool_final.txt").read_text().split())

    def blocks(self, c):
        b = pd.read_csv(self.e / "blocks.csv")
        b = b[b.collection == c]
        return dict(zip(b.file, b.block))

    def labels_external(self, c):
        key = ("L18", c, self.sealed)
        if key not in self._cache:
            parts = [pd.read_parquet(self.e / c / "open" / "labels_calibration.parquet")]
            if not self.sealed:
                parts.append(pd.read_parquet(heldout_path(self.e, c)))
            L = pd.concat(parts)
            self._cache[key] = L[L.cls >= 0][["file", "cls"] + PC.BOX].reset_index(drop=True)
        return self._cache[key]

    def pred_external(self, run, c):
        return pd.read_parquet(self.p / run / f"{c}.parquet", columns=PC.PRED_COLS)


PROCEDURE_COLS = ["track", "detector", "collection", "class_set", "arm", "draw", "threshold", "stated_recall",
                  "stated_lo", "stated_hi", "decision", "pilot_boxes", "pilot_blocks", "pilot_shortfall", "n", "M",
                  "recall_floor"]


def eligibility(a):
    return json.load(open(Path(a.ext18) / "exp18_eligibility.json"))


def chains(a, params, R, cons, procedure_only=False):
    """Every eligible network and endpoint: the selected checkpoint, then the sensitivity checkpoints of phase B
    (setup A on every endpoint; setup B on the merged-crack endpoint only, without the benchmark floors, which are
    defined on setup A's folds). One draw, constants for an unseen network, streams keyed by the network's index."""
    from pr_exp1 import run_chain
    rep = eligibility(a)
    out = []
    for i, (s, v) in enumerate(sorted(rep["networks"].items())):
        for track, ok in v["eligible"].items():
            if not ok:
                continue
            cs = TRACK_CLASS_SET[track]; u = cons["values"][cs]["unseen"]
            dets = [(params["selected"], True)] + [(d, True) for d in a.extra_detectors]
            if track == "merged_crack":
                dets += [(d, False) for d in a.setup_b]
            for det, floors in dets:
                D = run_chain(R, s, cs, det, str(Path(a.ext18) / "rules_split"), params, u["n"], u["M"],
                              PC.recall_floor(cons["recall_floor"], cs), 100 + i, 1, procedure_only=procedure_only,
                              floors=floors)
                b = D.attrs.pop("boot", None); D.attrs = {}
                if b is not None:
                    a.boots.append(b.assign(track=track))
                D.insert(0, "track", track)
                out.append(D)
    return out


def decide(a, params):
    """Phase C, step 1: threshold, pilot estimate and decision from calibration labels only."""
    import yaml
    PC.locked_scripts(a.root)
    cons = yaml.safe_load(open(a.constants))
    out = chains(a, params, Root18(a.root, a.ext18, a.pred, sealed=True), cons, procedure_only=True)
    if not out:
        PC.log("no eligible network"); PC.jdump(dict(eligible=0), Path(a.out) / "e18_none.json"); return
    D = pd.concat(out, ignore_index=True)
    D = D[D.arm == "procedure"][[c for c in PROCEDURE_COLS if c in D.columns]]
    D.to_csv(Path(a.out) / "e18_decisions.csv", index=False)
    ad = sorted(D[(D.decision == "adapt") & (D.class_set == "four_class") & (D.detector == params["selected"])]
                .collection.unique())
    (Path(a.out) / "adapt_networks.txt").write_text("\n".join(ad) + ("\n" if ad else ""))
    PC.log(f"decisions written; adaptation needed for: {ad or 'none'}")


def adapt(a, params):
    """Phase C, step 2 (graphics card), only for networks where the selected checkpoint's four-class decision was
    adapt: fine-tune with the experiment 5 recipe on adapt_size calibration images (whole blocks in a seeded order,
    pilot blocks first), keep a fresh audit set of further blocks apart, and predict the calibration and held-out
    parts. Only calibration labels are read."""
    import pr_finetune as F
    PC.locked_scripts(a.root)
    rep = eligibility(a)
    R = Root18(a.root, a.ext18, a.pred, sealed=True); p5 = params["exp5"]; n = int(params["exp1"]["adapt_size"])
    order_ix = {s_: i for i, s_ in enumerate(sorted(rep["networks"]))}
    sdir = Path(a.out) / "samples"; sdir.mkdir(parents=True, exist_ok=True)
    for slug in a.slugs:
        cal, held = PC.read_split(str(Path(a.ext18) / "rules_split"), slug)
        B = R.blocks(slug); L = R.labels_external(slug)
        nbox = L.groupby("file").size().to_dict()
        members = {}
        for f in cal:
            members.setdefault(B[f], []).append(f)
        order = F.block_order(cal, B, params["seed"], 100 + order_ix[slug], 0)
        pilot, audit, i, cnt = [], [], 0, 0
        while i < len(order) and cnt < p5["pilot_boxes"]:
            pilot.append(order[i]); cnt += sum(nbox.get(f, 0) for f in members[order[i]]); i += 1
        cnt = 0
        while i < len(order) and cnt < p5["audit_boxes"]:
            audit.append(order[i]); cnt += sum(nbox.get(f, 0) for f in members[order[i]]); i += 1
        train = []
        for b in pilot + order[i:]:
            if len(train) >= n:
                break
            train += members[b]
        aud = [f for b in audit for f in members[b]]
        assert not set(train) & set(aud)
        I = pd.read_csv(Path(a.ext18) / slug / "pool" / "images.csv").set_index("file")
        S = I[["width", "height"]].rename(columns={"width": "src_w", "height": "src_h"})
        y = F.yolo_dataset(R, slug, train, a.ext18, a.work, f"ds18_{slug}", L=L, S=S)
        w, secs = F.finetune(Path(a.weights_dir) / f"{params['selected']}_best.pt", y, p5["recipe"], a.work,
                             f"e18_{slug}", a.device)
        wdst = Path(a.out) / "weights" / f"e18_{slug}.pt"; wdst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(w, wdst)
        imgp = lambda fs: [str(Path(a.ext18) / slug / "pool" / "images" / f) for f in fs]  # noqa: E731
        F.predict({f"adapted18_{slug}": str(wdst)}, {f"{slug}_heldout": imgp(held), f"{slug}_calibration": imgp(cal)},
                  a, Path(a.out) / "pred")
        PC.jdump(dict(network=slug, train_images=train, audit_images=aud, shortfall=len(train) < n,
                      weights_sha256=PC.sha(wdst), train_seconds=round(secs, 1)), sdir / f"{slug}.json")
        PC.log(f"{slug}: adapted in {secs:.0f} s")


def commit(a, params):
    """Phase C, step 3: exp18_commitments.json. Per network, endpoint and checkpoint: threshold, stated recall with
    its interval, decision, sample sizes; for adapted networks the weights hash, the threshold recalibrated on the
    calibration images not used for training, and the fresh audit estimate with its interval. Calibration labels
    only."""
    import yaml
    PC.locked_scripts(a.root)
    cons = yaml.safe_load(open(a.constants))
    D = pd.read_csv(a.decisions)
    k = params["class_sets"]["four_class"]["k"]
    zd = __import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + params["exp3"]["decision_level"] / 2)
    R = Root18(a.root, a.ext18, a.pred, sealed=True)
    adapted = []
    for smp_p in sorted((Path(a.adapted) / "samples").glob("*.json")) if a.adapted else []:
        smp = json.load(open(smp_p)); slug = smp["network"]
        cal, _ = PC.read_split(str(Path(a.ext18) / "rules_split"), slug)
        Pc = pd.read_parquet(Path(a.adapted) / "pred" / f"adapted18_{slug}" / f"{slug}_calibration.parquet",
                             columns=PC.PRED_COLS)
        rest = sorted(set(cal) - set(smp["train_images"]))
        pc = Pc[Pc.file.isin(set(rest)) & Pc.cls.isin(PC.FOUR)].conf.sort_values(ascending=False).to_numpy()
        t = PC.threshold_for_rate(pc, len(rest), k) if rest else float("nan")
        rec = dict(network=slug, track="four_class", detector=f"adapted18_{slug}", weights_sha256=smp["weights_sha256"],
                   train_images=len(smp["train_images"]), audit_images=len(smp["audit_images"]),
                   recalibrated_threshold=float(t))
        if np.isfinite(t) and smp["audit_images"]:
            B = R.blocks(slug)
            TA = PC.ThresholdRun(Pc, R.labels_external(slug), smp["audit_images"], PC.FOUR)
            _, m, x = TA.per_image_at(t)
            ab = pd.DataFrame(dict(b=[B[f] for f in TA.files], m=m, x=x)).groupby("b").sum()
            est, lo, hi, _ = PC.ratio_ci(ab.m.to_numpy(), ab.x.to_numpy(), zd)
            rec.update(audit_estimate=float(est), audit_lo=float(lo), audit_hi=float(hi), audit_boxes=int(x.sum()),
                       decision="accept" if lo >= PC.recall_floor(cons["recall_floor"], "four_class") else "stop")
        adapted.append(rec)
    recs = [{k: (None if isinstance(v, float) and not np.isfinite(v) else (v.item() if hasattr(v, "item") else v))
             for k, v in r.items()} for r in D.to_dict("records")]  # full float precision, as read from the csv
    out = dict(procedure=recs, adapted=adapted,
               inputs=dict(constants_sha256=PC.sha(a.constants), decisions_sha256=PC.sha(a.decisions),
                           prediction_manifest_sha256=PC.sha(Path(a.pred) / "MANIFEST.sha256"),
                           eligibility_sha256=PC.sha(Path(a.ext18) / "exp18_eligibility.json"),
                           split_lists_sha256={p.name: PC.sha(p) for p in
                                               sorted((Path(a.ext18) / "rules_split" / "split").glob("*.txt"))}))
    PC.jdump(out, Path(a.out) / "exp18_commitments.json")
    PC.log(f"commitments written: {len(D)} procedure rows, {len(adapted)} adapted networks")


def score(a, params):
    """Phase D: held-out labels opened after their hashes are checked; every committed row scored."""
    import io, yaml
    PC.locked_scripts(a.root)
    rep = eligibility(a)
    for s, v in rep["networks"].items():
        if PC.sha(heldout_path(a.ext18, s)) != v["heldout_labels_sha256"]:
            sys.exit(f"STOP: held-out labels of {s} changed since phase A")
    com = json.load(open(a.commitments))
    cons = yaml.safe_load(open(a.constants))
    R = Root18(a.root, a.ext18, a.pred)
    out = chains(a, params, R, cons)
    if out:  # the procedure part must equal the committed phase C decisions
        D0 = pd.DataFrame(com["procedure"])
        D0 = pd.read_csv(io.StringIO(D0.to_csv(index=False)))
        D1 = pd.concat(out, ignore_index=True)
        D1 = D1[D1.arm == "procedure"][list(D0.columns)].reset_index(drop=True)
        D1 = pd.read_csv(io.StringIO(D1.to_csv(index=False)))
        try:
            pd.testing.assert_frame_equal(D0, D1, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        except AssertionError as e:
            PC.log(str(e))
            sys.exit("STOP: the procedure recomputed in phase D differs from the committed decisions")
    k = params["class_sets"]["four_class"]["k"]
    rows = []
    for ad in com["adapted"]:
        slug = ad["network"]; t = ad["recalibrated_threshold"]
        _, held = PC.read_split(str(Path(a.ext18) / "rules_split"), slug)
        row = dict(track="four_class", detector=ad["detector"], collection=slug, class_set="four_class", arm="adapted",
                   threshold=t, stated_recall=ad.get("audit_estimate"), stated_lo=ad.get("audit_lo"),
                   stated_hi=ad.get("audit_hi"), decision=ad.get("decision"),
                   recall_floor=PC.recall_floor(cons["recall_floor"], "four_class"))
        if t is not None and np.isfinite(t):
            Ph = pd.read_parquet(Path(a.adapted) / "pred" / ad["detector"] / f"{slug}_heldout.parquet",
                                 columns=PC.PRED_COLS)
            T = PC.ThresholdRun(Ph, R.labels_external(slug), held, PC.FOUR); al, m, x = T.per_image_at(t)
            row.update(true_recall=m.sum() / x.sum() if x.sum() else np.nan, heldout_rate=100 * al.sum() / len(held))
            row["rate_rel_error"] = row["heldout_rate"] / k - 1
            if row["stated_recall"] is not None:
                row["recall_error"] = row["stated_recall"] - row["true_recall"]
        rows.append(row)
    if rows:
        out.append(pd.DataFrame(rows))
    if not out:
        PC.log("no eligible network: experiment 18 reports the screening only")
        PC.jdump(dict(eligible=0), Path(a.out) / "e18_none.json"); return
    D = pd.concat(out, ignore_index=True)
    crit, marg = params["exp2"]["criterion_rel"], params["exp3"]["margin"]
    D["rate_within"] = D.rate_rel_error.abs() <= crit
    D["recall_within"] = D.recall_error.abs() <= marg
    D["role"] = np.where(D.detector == params["selected"], "selected",
                         np.where(D.arm == "adapted", "selected_adapted", "sensitivity"))
    D.to_csv(Path(a.out) / "e18_results.csv", index=False)
    if a.boots:
        pd.concat(a.boots, ignore_index=True).to_parquet(Path(a.out) / "e18_bootstrap.parquet")
    P = D[D.arm.isin(["procedure", "adapted"]) & D.role.isin(["selected", "selected_adapted"])]
    failed = sorted({f"{r.collection} ({r.track}, {r.arm})" for r in P.itertuples()
                     if not (r.rate_within and r.recall_within)})
    PC.jdump(dict(networks=int(P.collection.nunique()), rows=len(P), within_rate=int(P.rate_within.sum()),
                  within_recall=int(P.recall_within.sum()), decisions=P.decision.value_counts().to_dict(),
                  criterion_failed=failed), Path(a.out) / "e18_summary.json")
    PC.log("experiment 18 scored")


def unseal_check(a):
    """The prediction manifest must exist and match the hash committed before scoring."""
    m = Path(a.pred) / "MANIFEST.sha256"
    if not m.exists():
        sys.exit("STOP: prediction manifest missing")
    if PC.sha(m) != a.pred_manifest_sha:
        sys.exit("STOP: prediction manifest differs from the hash committed before scoring")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "prepare", "overlap", "decide", "adapt", "commit", "score"])
    ap.add_argument("--raw", default="/workspace/raw18", help="download and extraction folder of the candidates")
    ap.add_argument("--root", required=True); ap.add_argument("--rules", required=True)
    ap.add_argument("--ext18", required=True); ap.add_argument("--candidates", default="")
    ap.add_argument("--c0", default="/workspace/run/inputs/c0"); ap.add_argument("--device", default="0")
    ap.add_argument("--hash_only_test", action="store_true", help="synthetic tests only: hashes, no descriptors")
    ap.add_argument("--out", default=""); ap.add_argument("--pred", default="")
    ap.add_argument("--constants", default=""); ap.add_argument("--extra_detectors", nargs="*", default=[])
    ap.add_argument("--pred_manifest_sha", default="")
    ap.add_argument("--slugs", nargs="*", default=[], help="adapt: networks where the procedure said adapt")
    ap.add_argument("--adapted", default="", help="score: output folder of the adapt step, if it ran")
    ap.add_argument("--decisions", default="", help="commit: e18_decisions.csv of phase C")
    ap.add_argument("--commitments", default="", help="score: the committed exp18_commitments.json")
    ap.add_argument("--commitments_sha", default="")
    ap.add_argument("--setup_b", nargs="*", default=[], help="setup B models (merged-crack endpoint only)")
    ap.add_argument("--work", default="/workspace/pr_work"); ap.add_argument("--weights_dir", default="/workspace/run/weights")
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--rules_sha", default=""); ap.add_argument("--params_sha", default="")
    ap.add_argument("--constants_sha", default="")
    ap.add_argument("--skip_gate", action="store_true", help="synthetic tests only")
    a = ap.parse_args()
    a.boots = []
    if not a.skip_gate:
        PC.verify_lock(a.root, a.manifest_sha); PC.rule_gate(a.rules, a.rules_sha, a.params_sha)
        if a.cmd in ("decide", "commit", "score"):
            unseal_check(a)
            if PC.sha(a.constants) != a.constants_sha:
                sys.exit("STOP: exp1_constants.yaml differs from the approved hash")
        if a.cmd == "score":
            if PC.sha(a.commitments) != a.commitments_sha:
                sys.exit("STOP: exp18_commitments.json differs from the committed hash")
            if a.adapted and not (Path(a.adapted) / "MANIFEST.sha256").exists():
                sys.exit("STOP: adaptation outputs have no manifest")
    params = PC.load_params(a.rules)
    PC.locked_scripts(a.root)
    Path(a.ext18).mkdir(parents=True, exist_ok=True)
    if a.cmd in ("decide", "adapt", "commit", "score"):
        Path(a.out).mkdir(parents=True, exist_ok=True)
    {"download": download, "prepare": prepare, "overlap": overlap, "decide": decide, "adapt": adapt, "commit": commit,
     "score": score}[a.cmd](a, params)
    if a.cmd in ("decide", "adapt", "commit", "score"):
        PC.write_manifest(a.out)
