#!/usr/bin/env python3
"""Experiment 18 input handling on a small synthetic release (tests only): a COCO release in two parts (train and
valid), rotated and reflected copies that must collapse to one image, and crops of one parent photograph that
must share a block. Usage: test_exp18_inputs.py FIXTURE_DIR (a folder made by make_exp18_fixture.py build)."""
import json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

S = Path(__file__).resolve().parents[1] / "scripts"


def main(fx):
    fx = Path(fx); rng = np.random.default_rng(7)
    raw = fx / "raw_coco"; out = fx / "ext18_coco"
    for d in (raw, out):
        if d.exists():
            subprocess.run(["rm", "-rf", str(d)], check=True)
    anns, imgs = {0: [], 1: []}, {0: [], 1: []}
    cats = [{"id": 1, "name": "D00"}, {"id": 2, "name": "D10"}, {"id": 3, "name": "D20"}, {"id": 4, "name": "D40"},
            {"id": 5, "name": "D50"}]
    aid = 0
    for part in (0, 1):
        d = raw / ("train" if part == 0 else "valid"); d.mkdir(parents=True)
        for i in range(12):
            base = Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)).resize((320, 240), Image.BICUBIC)
            name = f"photo{part}{i:02d}_crop{i % 3}.jpg"
            base.save(d / name, quality=95)
            items = [(name, base)]
            if part == 0 and i < 3:   # released rotated and reflected copies
                for k, t in enumerate((Image.Transpose.ROTATE_180, Image.Transpose.FLIP_LEFT_RIGHT)):
                    n2 = f"photo{part}{i:02d}_aug{k}.jpg"; base.transpose(t).save(d / n2, quality=95)
                    items.append((n2, base.transpose(t)))
            if part == 1 and i < 2:   # export-tool copies named after the source image (brightness changed)
                from PIL import ImageEnhance
                b2 = ImageEnhance.Brightness(base).enhance(1.4)
                n2 = name.replace(".jpg", "_jpg.rf.copy1.jpg"); b2.save(d / n2, quality=95)
                items.append((n2, b2))
            for n, im in items:
                iid = len(imgs[part]) + 1
                imgs[part].append({"id": iid, "file_name": n, "width": im.width, "height": im.height})
                for _ in range(2):
                    aid += 1
                    anns[part].append({"id": aid, "image_id": iid, "category_id": int(rng.integers(1, 6)),
                                       "bbox": [float(rng.uniform(0, 200)), float(rng.uniform(0, 150)), 40.0, 30.0]})
        json.dump({"images": imgs[part], "annotations": anns[part], "categories": cats},
                  open(d / "_annotations.coco.json", "w"))
    cand = pd.DataFrame([dict(slug="gamma", release="1", team="gamma team", format="coco",
                              images_dir="train;valid", labels_dir="train/_annotations.coco.json;valid/_annotations.coco.json",
                              class_names="", license="CC BY 4.0", url="https://example.org", download_urls="file:///test",
                              sha256="test", parent_regex=r"(photo\d)\d{2}_crop",
                              copy_regex=r"(photo1\d\d_crop\d)(?:_jpg\.rf\.\w+)?\.jpg$")])
    cand_f = fx / "cand_coco.csv"; cand.to_csv(cand_f, index=False)
    # the reader resolves relative folders under <raw>/<slug>/files
    (fx / "rawroot" / "gamma").mkdir(parents=True, exist_ok=True)
    link = fx / "rawroot" / "gamma" / "files"
    if not link.exists():
        link.symlink_to(raw.resolve())
    X = ["--root", str(fx / "root"), "--rules", str(fx / "rules"), "--ext18", str(out), "--candidates", str(cand_f),
         "--raw", str(fx / "rawroot"), "--skip_gate"]
    subprocess.run([sys.executable, str(S / "pr_exp18.py"), "prepare", *X], check=True)
    P = json.load(open(out / "exp18_prepare.json"))["gamma"]
    assert P["images"] == 32, P["images"]                       # 24 photos + 6 rotated or reflected + 2 named copies
    assert P["unmapped_by_raw_class"].get("D50", 0) > 0
    I = pd.read_csv(out / "gamma" / "pool" / "images.csv")
    assert I.source.str.startswith("part0/").sum() == 18 and I.source.str.startswith("part1/").sum() == 14
    subprocess.run([sys.executable, str(S / "pr_exp18.py"), "overlap", *X, "--c0", str(fx / "c0"), "--device", "cpu",
                    "--hash_only_test"], check=True)
    E = json.load(open(out / "exp18_eligibility.json"))["networks"]["gamma"]
    assert E["transformed_copies_dropped"] == 8 and E["transformed_copy_groups"] == 5, E
    B = pd.read_csv(out / "blocks.csv")
    B["parent"] = B.file.str.extract(r"(photo\d)\d{2}_crop")[0]
    assert (B.dropna().groupby("parent").block.nunique() == 1).all()   # crops of one parent share a block
    assert E["images_released"] == 32 and E["collapse_failed"] is False, E
    # a stated kept-share limit below what the collapse keeps (24 of 32) drops the network from every endpoint
    cand2 = cand.assign(max_kept_share="0.5"); cand2_f = fx / "cand_coco_limit.csv"; cand2.to_csv(cand2_f, index=False)
    out2 = fx / "ext18_coco_limit"
    X2 = [v if v != str(out) else str(out2) for v in X]; X2[X2.index(str(cand_f))] = str(cand2_f)
    for ph, extra in (("prepare", []), ("overlap", ["--c0", str(fx / "c0"), "--device", "cpu", "--hash_only_test"])):
        subprocess.run([sys.executable, str(S / "pr_exp18.py"), ph, *X2, *extra], check=True)
    E2 = json.load(open(out2 / "exp18_eligibility.json"))["networks"]["gamma"]
    assert E2["collapse_failed"] is True and not any(E2["eligible"].values()), E2
    # "auto" layout: every COCO file found is one part; a stray image and a non-COCO JSON file are ignored
    Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(raw / "train" / "stray.jpg")
    json.dump({"note": "not a COCO file"}, open(raw / "README.json", "w"))
    cand3 = cand.assign(images_dir="auto", labels_dir="auto"); cand3_f = fx / "cand_coco_auto.csv"
    cand3.to_csv(cand3_f, index=False); out3 = fx / "ext18_coco_auto"
    X3 = [v if v != str(out) else str(out3) for v in X]; X3[X3.index(str(cand_f))] = str(cand3_f)
    subprocess.run([sys.executable, str(S / "pr_exp18.py"), "prepare", *X3], check=True)
    P3 = json.load(open(out3 / "exp18_prepare.json"))["gamma"]
    assert P3["images"] == 32 and len(P3["parts"]) == 2 and P3["images_not_in_any_coco_file"] == 1, P3
    assert [q["images_dir"] for q in P3["parts"]] == ["train", "valid"], P3["parts"]
    I3 = pd.read_csv(out3 / "gamma" / "pool" / "images.csv")
    assert sorted(I3.file) == sorted(I.file), "auto layout reads different images"
    L3 = pd.read_parquet(out3 / "gamma" / "pool" / "labels.parquet")
    assert len(L3) == sum(P["boxes_by_class"].values()) + int(sum(P["unmapped_by_raw_class"].values()))
    print("experiment 18 input test passed:", {k: E[k] for k in ("transformed_copies_dropped", "parent_links", "blocks")})


if __name__ == "__main__":
    main(sys.argv[1])
