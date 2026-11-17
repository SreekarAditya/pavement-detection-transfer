#!/usr/bin/env python3
"""Experiment 11: what the review queue contains (task 05).

  a      Part A from labels: each unmatched alert of the selected detector at the four-class quota is classed as
         wrong subclass, near miss, unmapped road object or background; agreement count of the other checkpoints
  b      Part B packets: stratified sample, two blind rater viewers (no confidence, class, checkpoint or reference box)
  c      merge two raters' answers, Cohen's kappa, adjudication packet; with --adjudication, the final table and the
         recall sensitivity
Unmapped boxes come from the prepared pools of scripts/pdt_s2_external.py (ext/<c>/pool/labels.parquet keeps
boxes of unmapped classes with cls = -1).
"""
import argparse, base64, io, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as PC  # noqa: E402

CATS = ["missing_mapped_distress", "other_distress", "road_object", "not_damage", "cannot_tell"]
SUBS = ["longitudinal crack", "transverse crack", "alligator crack", "pothole"]
ALERT_OUTLINE = (255, 0, 255)
CROP_WINDOW = ["crop_left", "crop_top", "crop_right", "crop_bottom"]
ADJUDICATION_SEED = 2028
AGREE = ["agree_with_first", "agree_with_second"]


def part_a(a, params, out):
    PC.locked_scripts(a.root)
    R = PC.Root(a.root)
    p = params["exp11"]; sel = params["selected"]; k = params["k"]["four_class"]
    C = sys.modules["pdt_core"]
    rows, labelled = [], {}
    for c in PC.EXT_FOUR:
        files = R.pool(c); B = R.blocks(c)
        L = R.labels_external(c)
        U = pd.read_parquet(Path(a.ext_root) / c / "pool" / "labels.parquet")
        U = U[(U.cls < 0) & U.file.isin(set(files))]
        labelled[c] = sorted(U.raw_class.astype(str).unique().tolist())
        Q = PC.top_quota(R.pred_external(sel, c), files, PC.FOUR, k)
        _, pm, _ = PC.match_flags(Q, L)
        Qu = Q[~pm].reset_index(drop=True)
        others = {r: PC.top_quota(R.pred_external(r, c), files, PC.FOUR, k) for r in PC.RUNS_A if r != sel}
        Lg, Ug = L.groupby("file").indices, U.groupby("file").indices
        Og = {r: (o, o.groupby("file").indices) for r, o in others.items()}
        Lb, Lc = L[PC.BOX].to_numpy(float), L.cls.to_numpy()
        Ub, Uc = U[PC.BOX].to_numpy(float), U.raw_class.astype(str).to_numpy()
        for i, q in Qu.iterrows():
            pb = np.array([[q.x1, q.y1, q.x2, q.y2]], float)
            li = Lg.get(q.file, []); ui = Ug.get(q.file, [])
            cls_a, near, unm = "background", 0.0, ""
            if len(li):
                iou = C.box_iou(Lb[li], pb)[:, 0]
                same = Lc[li] == q.cls
                if (iou[~same] >= 0.5).any():
                    cls_a = "wrong_subclass"
                elif same.any() and 0.2 <= iou[same].max() < 0.5:
                    cls_a, near = "near_miss", float(iou[same].max())
            if cls_a == "background" and len(ui):
                # share of the alert's own area inside an unmapped object (patches and manholes are often much
                # larger than the alert, so overlap is measured on the alert, not as intersection over union)
                ub = Ub[ui]
                iw = np.clip(np.minimum(ub[:, 2], pb[0, 2]) - np.maximum(ub[:, 0], pb[0, 0]), 0, None)
                ih = np.clip(np.minimum(ub[:, 3], pb[0, 3]) - np.maximum(ub[:, 1], pb[0, 1]), 0, None)
                ioa = iw * ih / max((pb[0, 2] - pb[0, 0]) * (pb[0, 3] - pb[0, 1]), 1e-9)
                if ioa.max() >= p["unmapped_overlap"]:
                    cls_a, unm = "unmapped_road_object", Uc[ui][int(ioa.argmax())]
            agree = 0
            for r, (o, og) in Og.items():
                oi = og.get(q.file, [])
                if len(oi):
                    ob = o.iloc[oi]
                    ob = ob[ob.cls == q.cls]
                    if len(ob) and (C.box_iou(ob[PC.BOX].to_numpy(float), pb)[:, 0] >= p["agreement_iou"]).any():
                        agree += 1
            rows.append(dict(collection=c, file=q.file, block=B[q.file], cls=int(q.cls), conf=float(q.conf),
                             x1=q.x1, y1=q.y1, x2=q.x2, y2=q.y2, part_a=cls_a, near_iou=near, unmapped_class=unm,
                             agreement=agree))
        PC.log(f"{c}: {len(Q)} alerts, {len(Qu)} unmatched")
    A = pd.DataFrame(rows)
    if p["strata"] == "tertiles":   # confidence tertiles of the unmatched alerts, within each collection
        A["stratum"] = A.groupby("collection").conf.transform(
            lambda x: pd.qcut(x.rank(method="first"), 3, labels=False)).astype(int)
    else:
        A["stratum"] = pd.cut(A.conf, p["strata"], right=False, labels=False)
    A.to_parquet(out / "e11a_alerts.parquet")
    srows = []
    for (c, s), g in list(A.groupby(["collection", "stratum"])) + [((c, "all"), g) for c, g in A.groupby("collection")]:
        for cat in ["wrong_subclass", "near_miss", "unmapped_road_object", "background"]:
            ind = (g.part_a == cat).astype(float).to_numpy()
            from pr_bench import boot_share
            bs = boot_share(ind, g.block.to_numpy(), params["bootstrap"]["draws"], params["seed"])
            srows.append(dict(collection=c, stratum=s, part_a=cat, alerts=len(g), share=float(ind.mean()), ci95=PC.pct(bs)))
    pd.DataFrame(srows).to_csv(out / "e11a_summary.csv", index=False)
    PC.jdump(dict(unmapped_classes_labelled=labelled), out / "e11a_unmapped_classes.json")


def sample_b(A, params):
    p = params["exp11"]; rng = np.random.default_rng(params["seed"])
    picks = []
    for c, g in A.groupby("collection"):
        strata = sorted(g.stratum.dropna().unique())
        n = p["sample_per_collection"]
        per = {s: n // len(strata) for s in strata}
        for s in strata[: n - sum(per.values())]:
            per[s] += 1
        short = 0; chosen = []
        for s in strata:
            idx = g.index[g.stratum == s].to_numpy()
            take = min(per[s], len(idx)); short += per[s] - take
            chosen += list(rng.choice(idx, take, replace=False))
        rest = np.setdiff1d(g.index.to_numpy(), chosen)
        if short and len(rest):
            chosen += list(rng.choice(rest, min(short, len(rest)), replace=False))
        picks += chosen
    S = A.loc[sorted(picks)].reset_index().rename(columns={"index": "alert_row"})
    S.insert(0, "item", [f"I{i:04d}" for i in rng.permutation(len(S))])
    return S.sort_values("item").reset_index(drop=True)


def encode_jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_axis(lo, hi, extent):
    size = min(extent, max(160, math.ceil(2 * (hi - lo))))
    start = round((lo + hi - size) / 2)
    start = min(max(start, 0), extent - size)
    return start, start + size


def rating_item_images(row, img_path):
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(img_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    full = image.copy()
    box = (row.x1, row.y1, row.x2, row.y2)
    stroke = max(3, round(max(image.size) / 400))
    ImageDraw.Draw(full).rectangle(box, outline=ALERT_OUTLINE, width=stroke)
    full.thumbnail((1280, 1280))

    left, right = crop_axis(row.x1, row.x2, image.width)
    top, bottom = crop_axis(row.y1, row.y2, image.height)
    window = (left, top, right, bottom)
    crop = image.crop(window)
    crop_box = (row.x1 - left, row.y1 - top, row.x2 - left, row.y2 - top)
    ImageDraw.Draw(crop).rectangle(crop_box, outline=ALERT_OUTLINE, width=stroke)
    return encode_jpeg(full), encode_jpeg(crop), window


VIEWER = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Rating packet</title>
<style>body{font-family:sans-serif;margin:16px}img{max-width:100%}.row{display:flex;gap:12px;flex-wrap:wrap}
button{margin:4px;padding:8px 12px;font-size:15px}#sub button{background:#eef}.done{color:#080}</style></head><body>
<h2 id="h"></h2><div class="row"><div><img id="full"></div><div><img id="crop"></div></div>
<dl id="prior" hidden><dt>First answer</dt><dd id="first"></dd><dt>Second answer</dt><dd id="second"></dd></dl>
<div id="cats"></div><div id="sub" style="display:none">Which class? </div>
<p><button onclick="prev()">Back</button> <span id="st"></span> <button onclick="save()">Save answers</button></p>
<script>
const RATER=__RATER__; const ITEMS=__ITEMS__; const CATS=__CATS__; const SUBS=__SUBS__;
let i=0, t0=Date.now(); const ans={};
function displayAnswer(a){return a.category.replaceAll('_',' ')+' / '+(a.subclass||'No subclass');}
function show(){
if(!ITEMS.length){document.getElementById('h').textContent='No items to adjudicate.';cd.hidden=true;return;}
const it=ITEMS[i];document.getElementById('h').textContent=`Item ${i+1} of ${ITEMS.length} (${it.item})`;
document.getElementById('prior').hidden=!it.first;
if(it.first){document.getElementById('first').textContent=displayAnswer(it.first);
document.getElementById('second').textContent=displayAnswer(it.second);}
document.getElementById('full').src='data:image/jpeg;base64,'+it.full;document.getElementById('crop').src='data:image/jpeg;base64,'+it.crop;
document.getElementById('sub').style.display='none';const a=ans[it.item];document.getElementById('st').textContent=a?('answered: '+a.category+(a.subclass?(' / '+a.subclass):'')):'';t0=Date.now();}
function rec(cat,sub){const it=ITEMS[i];const prev=ans[it.item];const s=(Date.now()-t0)/1000+(prev?prev.seconds:0);
ans[it.item]={category:cat,subclass:sub||null,seconds:s};if(i<ITEMS.length-1){i++;show();}else{show();alert('Last item done. Press Save answers.');}}
function prev(){if(i>0){i--;show();}}
function save(){const b=new Blob([JSON.stringify({rater:RATER,answers:ans},null,1)],{type:'application/json'});
const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='answers_'+RATER+'.json';a.click();}
const cd=document.getElementById('cats');CATS.forEach(c=>{const b=document.createElement('button');b.textContent=c.replaceAll('_',' ');
b.onclick=()=>{if(c==='missing_mapped_distress'){document.getElementById('sub').style.display='block';}else rec(c);};cd.appendChild(b);});
const sd=document.getElementById('sub');SUBS.forEach(s=>{const b=document.createElement('button');b.textContent=s;b.onclick=()=>rec('missing_mapped_distress',s);sd.appendChild(b);});
show();
</script></body></html>"""


def viewer(path, rater, items, cats):
    html = (VIEWER.replace("__RATER__", json.dumps(rater)).replace("__ITEMS__", json.dumps(items))
            .replace("__CATS__", json.dumps(cats)).replace("__SUBS__", json.dumps(SUBS)))
    Path(path).write_text(html)


def part_b(a, params, out):
    A = pd.read_parquet(out / "e11a_alerts.parquet")
    S = sample_b(A, params)
    priv = out / "private"; priv.mkdir(exist_ok=True)
    imgs = {}
    windows = []
    for r in S.itertuples():
        p = Path(a.ext_root) / r.collection / "pool" / "images" / r.file
        full, crop, window = rating_item_images(r, p)
        imgs[r.item] = (full, crop)
        windows.append(window)
    S[CROP_WINDOW] = pd.DataFrame(windows, index=S.index, columns=CROP_WINDOW, dtype="int64")
    S.to_parquet(priv / "e11b_key.parquet")
    for rater, seed in (("Aditya", params["seed"]), ("Sahasra", params["seed"] + 1)):
        order = np.random.default_rng(seed).permutation(len(S))
        items = [dict(item=S["item"].iloc[j], full=imgs[S["item"].iloc[j]][0], crop=imgs[S["item"].iloc[j]][1]) for j in order]
        d = out / f"packet_{rater}"; d.mkdir(exist_ok=True)
        viewer(d / "index.html", rater, items, CATS)
    (out / "RATER_INSTRUCTIONS.md").write_text(
        "# Rating the review queue\n\nEach item shows a road image and a close-up. Both views mark the same area with a magenta outline.\n"
        "Choose one answer for the boxed area:\n\n- missing mapped distress (then choose longitudinal crack, transverse "
        "crack, alligator crack or pothole)\n- other distress\n- road object (patch, manhole, marking, joint)\n- not damage\n"
        "- cannot tell\n\nWork alone and do not discuss any item until both raters are done. When finished, press Save "
        "answers and send the file.\n")
    PC.log(f"packets: {len(S)} items")


def kappa(x, y, cats):
    i = {c: j for j, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for u, v in zip(x, y):
        M[i[u], i[v]] += 1
    n = M.sum(); po = np.trace(M) / n; pe = (M.sum(0) * M.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan, M


def load_answers(path, items, categories):
    def unique_keys(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            sys.exit("STOP: duplicate key in an answer export")
        return obj

    document = json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)
    if not isinstance(document, dict) or not isinstance(document.get("rater"), str):
        sys.exit("STOP: malformed answer export")
    answers = document.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(items):
        sys.exit("STOP: answer export does not cover exactly the required items")
    for answer in answers.values():
        if not isinstance(answer, dict) or set(answer) != {"category", "subclass", "seconds"}:
            sys.exit("STOP: malformed answer")
        category, subclass, seconds = answer["category"], answer["subclass"], answer["seconds"]
        if category not in categories or (subclass not in SUBS if category == CATS[0] else subclass is not None):
            sys.exit("STOP: invalid answer category or subclass")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            sys.exit("STOP: invalid answer time")
    return document


def rating_images(path, items):
    html = Path(path).read_text()
    _, start, data = html.partition("const ITEMS=")
    payload, end, _ = data.partition("; const CATS=")
    if not start or not end:
        sys.exit("STOP: could not find the rebuilt rating packet images")
    rows = json.loads(payload)
    images = {r["item"]: {"full": r["full"], "crop": r["crop"]} for r in rows}
    if len(images) != len(rows) or set(images) != set(items):
        sys.exit("STOP: the rebuilt rating packet does not match the item key")
    return images


def part_c(a, params, out):
    private = out / "private"
    S = pd.read_parquet(private / "e11b_key.parquet")
    if S["item"].duplicated().any() or not set(CROP_WINDOW).issubset(S.columns):
        sys.exit("STOP: a unique item key from the rebuilt rating packets is required")
    r1, r2 = (load_answers(path, S["item"], CATS) for path in a.answers)
    if r1["rater"] == r2["rater"]:
        sys.exit("STOP: two distinct rater exports are required")
    x = [r1["answers"][i]["category"] for i in S["item"]]; y = [r2["answers"][i]["category"] for i in S["item"]]
    k, M = kappa(x, y, CATS)
    rng = np.random.default_rng(params["seed"])
    bs = [kappa(np.array(x)[s], np.array(y)[s], CATS)[0] for s in rng.integers(0, len(x), (params["bootstrap"]["draws"], len(x)))]
    PC.jdump(dict(kappa=k, ci95=PC.pct(bs), raters=[r1["rater"], r2["rater"]]), out / "e11c_kappa.json")
    pd.DataFrame(M, index=CATS, columns=CATS).to_csv(out / "e11c_confusion.csv")
    S["rater1"] = x; S["rater2"] = y
    S["sub1"] = [r1["answers"][i]["subclass"] for i in S["item"]]; S["sub2"] = [r2["answers"][i]["subclass"] for i in S["item"]]
    S["sec1"] = [r1["answers"][i]["seconds"] for i in S["item"]]; S["sec2"] = [r2["answers"][i]["seconds"] for i in S["item"]]
    eff = [dict(rater=r, median_seconds=float(np.median(S[c])), q25=float(np.percentile(S[c], 25)),
                q75=float(np.percentile(S[c], 75))) for r, c in ((r1["rater"], "sec1"), (r2["rater"], "sec2"))]
    pd.DataFrame(eff).to_csv(out / "e11c_effort.csv", index=False)
    dis = S[S.rater1 != S.rater2].sort_values("item")
    both_missing = S[(S.rater1 == CATS[0]) & (S.rater2 == CATS[0])]
    matching = int((both_missing.sub1 == both_missing.sub2).sum())
    pd.DataFrame([dict(status="unadjudicated", both_missing_mapped_distress=len(both_missing),
                       matching_subclasses=matching,
                       matching_share=matching / len(both_missing) if len(both_missing) else None)
                  ]).to_csv(out / "e11c_subclass_agreement_unadjudicated.csv", index=False)

    mapping_path = private / "e11c_answer_order.json"
    answer_hashes = [PC.sha(path) for path in a.answers]
    raters = (r1["answers"], r2["answers"])
    if not a.adjudication:
        if not a.rating_packet:
            sys.exit("STOP: the rebuilt rating packet is required for adjudication images")
        images = rating_images(a.rating_packet, S["item"])
        order_rng = np.random.default_rng(ADJUDICATION_SEED)
        order = order_rng.permutation(len(dis))
        first_rater = {item: int(index) for item, index in zip(dis["item"], order_rng.integers(0, 2, len(dis)))}
        items = []
        for j in order:
            item = dis["item"].iloc[j]
            first = first_rater[item]
            shown = [{key: raters[index][item][key] for key in ("category", "subclass")}
                     for index in (first, 1-first)]
            items.append(dict(item=item, **images[item], first=shown[0], second=shown[1]))
        d = out / "packet_adjudication"; d.mkdir(exist_ok=True)
        viewer(d / "index.html", "adjudicator", items, CATS + AGREE)
        PC.jdump(dict(seed=ADJUDICATION_SEED, answer_hashes=answer_hashes, first_rater=first_rater), mapping_path)
        S.to_parquet(private / "e11b_merged.parquet")
        PC.log("adjudication packet ready")
        return

    mapping = json.loads(mapping_path.read_text())
    first_rater = mapping["first_rater"]
    if (mapping["seed"] != ADJUDICATION_SEED or mapping["answer_hashes"] != answer_hashes
            or set(first_rater) != set(dis["item"])
            or any(type(index) is not int or index not in (0, 1) for index in first_rater.values())):
        sys.exit("STOP: the private answer order does not match these exports")
    adj = load_answers(a.adjudication, dis["item"], CATS + AGREE)["answers"]
    final, subclasses = [], []
    for _, r in S.iterrows():
        if r.rater1 == r.rater2:
            final.append(r.rater1)
            subclasses.append(r.sub1 if r.sub1 == r.sub2 else None)
        else:
            item = r["item"]
            answer = adj[item]
            if answer["category"] in AGREE:
                first = first_rater[item]
                answer = raters[first if answer["category"] == AGREE[0] else 1-first][item]
            final.append(answer["category"])
            subclasses.append(answer["subclass"])
    S["final"] = final
    S["final_subclass"] = subclasses
    S.to_parquet(out / "e11b_final.parquet")
    A = pd.read_parquet(out / "e11a_alerts.parquet")
    PC.locked_scripts(a.root); R = PC.Root(a.root)
    rows = []
    for c in PC.EXT_FOUR:
        g = A[A.collection == c]; s = S[S.collection == c]
        files = R.pool(c)
        Q = PC.top_quota(R.pred_external(params["selected"], c), files, PC.FOUR, params["k"]["four_class"])
        lm, pm, _ = PC.match_flags(Q, R.labels_external(c))
        matched, refs = int(lm.sum()), int(len(lm))
        est = 0.0
        for st, gs in g.groupby("stratum"):
            ss = s[s.stratum == st]
            if len(ss):
                est += len(gs) * float((ss.final == "missing_mapped_distress").mean())
        for strat, sub in (("all", s), ("agreement>=half", s[s.agreement >= len(PC.RUNS_A) // 2])):
            rows.append(dict(collection=c, stratum=strat, sampled=len(sub),
                             share_missing_mapped=float((sub.final == "missing_mapped_distress").mean()) if len(sub) else None,
                             share_any_distress=float(sub.final.isin(["missing_mapped_distress", "other_distress"]).mean())
                             if len(sub) else None))
        rows.append(dict(collection=c, stratum="recall_sensitivity", recall_registered=matched / refs if refs else None,
                         est_missing_damage_in_queue=est,
                         recall_with_missing_counted=(matched + est) / (refs + est) if refs + est else None,
                         useful_alert_share=(matched + est) / len(Q) if len(Q) else None))
    pd.DataFrame(rows).to_csv(out / "e11c_sensitivity.csv", index=False)
    PC.log("experiment 11 merged")


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.skip_gate:
        PC.verify_lock(a.root, a.manifest_sha)
        PC.rule_gate(a.rules, a.rules_sha, a.params_sha)
    params = PC.load_params(a.rules)
    {"a": part_a, "b": part_b, "c": part_c}[a.part](a, params, out)
    PC.write_manifest(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["a", "b", "c"])
    ap.add_argument("--root", required=True); ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ext_root", required=True)
    ap.add_argument("--answers", nargs=2, default=None); ap.add_argument("--adjudication", default="")
    ap.add_argument("--rating-packet", help="rebuilt 05B viewer whose image bytes are reused")
    ap.add_argument("--manifest_sha", default="ef49673bdc6301009223fb7eecd0aa71be7f53eb6220395d30262b64b19228c7")
    ap.add_argument("--rules_sha", default=""); ap.add_argument("--params_sha", default="")
    ap.add_argument("--skip_gate", action="store_true", help="synthetic tests only")
    main(ap.parse_args())
