#!/usr/bin/env python3
"""Gate G3, laptop part 1: build the viewing page for the near-duplicate boundary check.

Run inside the extracted g3_pairs folder (or pass --pairs). Writes g3_check.html next to manifest.json; open it
in a browser. The page shows each pair side by side, without similarity values or bin names, in one random order
(seed 2026), and records "same location", "different location" or "can't tell".

Serving rule (protocol Section 3): bins with fewer than 5 pairs in total are merged into the next higher bin
(groups). Each group's first 20 exported pairs (all, if fewer) form the sample, shown in one shuffled order across
groups. When the sample is finished, any group with fewer than 10 decided answers is given further pairs from its
reserve, in the exported order, until it has 10 decided or the reserve is used up.

Answers are saved in the browser as they are given and can be downloaded as g3_answers.json.
"""
import argparse, json
from pathlib import Path

import numpy as np

SEED, SAMPLE, MIN_DECIDED, MERGE_BELOW = 2026, 20, 10, 5


def groups_from_manifest(m):
    groups, carry = [], []
    for k, b in enumerate(m["bins"]):
        carry.append(b)
        last = k == len(m["bins"]) - 1
        if sum(x["n_pairs_total"] for x in carry) >= MERGE_BELOW or last:
            if sum(x["n_pairs_total"] for x in carry) > 0 or last:
                groups.append(carry)
            carry = []
    out = []
    for g in groups:
        sample, reserve = [], []
        for b in g:
            p = b["pairs"]
            sample += p[:SAMPLE]; reserve += p[SAMPLE:]
        out.append(dict(group="+".join(b["bin"] for b in g), bins=[b["bin"] for b in g],
                        n_pairs_total=sum(b["n_pairs_total"] for b in g), sample=sample, reserve=reserve))
    return out


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>G3 pair check</title>
<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;background:#111;color:#eee}
header{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;background:#1d1d1d}
#imgs{display:flex;gap:8px;padding:8px;height:calc(100vh - 170px)}
#imgs div{flex:1;display:flex;align-items:center;justify-content:center;background:#000}
#imgs img{max-width:100%;max-height:100%;cursor:zoom-in}
#bar{display:flex;gap:12px;justify-content:center;padding:12px}
button{font-size:18px;padding:12px 22px;border:0;border-radius:6px;cursor:pointer}
.same{background:#2e7d32;color:#fff}.diff{background:#c62828;color:#fff}.cant{background:#616161;color:#fff}
.small{font-size:14px;padding:8px 14px;background:#333;color:#eee}
#help{font-size:13px;color:#aaa;text-align:center;padding-bottom:8px}
#done{display:none;padding:40px;text-align:center}
</style></head><body>
<header><div><b>G3 pair check</b> &nbsp; <span id="prog"></span></div>
<div><button class="small" onclick="back()">Back</button> <button class="small" onclick="save()">Download answers</button></div></header>
<div id="main">
<div id="imgs"><div><img id="A"></div><div><img id="B"></div></div>
<div id="bar"><button class="same" onclick="ans('same')">Same location (S)</button>
<button class="diff" onclick="ans('different')">Different location (D)</button>
<button class="cant" onclick="ans('cant_tell')">Can't tell (C)</button></div>
<div id="help">Same location: the same stretch of pavement is visible in both (same crack shapes, patches, fixtures, markings, road edge or background objects), for example nearby frames of one drive. Different location: different roads that only look alike. Click an image to open it full size.</div>
</div>
<div id="done"><h2>All pairs answered.</h2><p>Click "Download answers" and send g3_answers.json.</p></div>
<script>
const G = __GROUPS__, ORDER = __ORDER__, KEY = "pdt_g3_answers_" + __MHASH__;
let st = null;
try { st = JSON.parse(localStorage.getItem(KEY)); } catch (e) {}
if (!st) st = {queue: ORDER.slice(), pos: 0, answers: {}, served: {}, extra: {}};
function persist(){ try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} }
function groupOf(id){ for (let g=0; g<G.length; g++){ if (G[g].ids.includes(id)) return g; } return -1; }
function decided(g){ let n=0; for (const id in st.answers){ if (groupOf(id)===g && st.answers[id]!=="cant_tell") n++; } return n; }
function refill(){
  // after the sample, top up groups below the minimum from their reserve, one pair at a time
  let added = false;
  for (let g=0; g<G.length; g++){
    const need = __MIN__ - decided(g);
    const used = st.extra[g] || 0;
    if (need > 0 && used < G[g].reserve.length){
      st.queue.push(G[g].reserve[used]); st.extra[g] = used + 1; added = true;
    }
  }
  return added;
}
function show(){
  if (st.pos >= st.queue.length && !refill()){
    document.getElementById("main").style.display="none"; document.getElementById("done").style.display="block";
    document.getElementById("prog").textContent = Object.keys(st.answers).length + " answered"; persist(); return;
  }
  const id = st.queue[st.pos];
  document.getElementById("A").src = "img/" + id + "_a.jpg";
  document.getElementById("B").src = "img/" + id + "_b.jpg";
  document.getElementById("prog").textContent = "pair " + (st.pos+1) + " of at least " + ORDER.length;
  persist();
}
function ans(a){ const id = st.queue[st.pos]; st.answers[id] = a; st.served[id] = new Date().toISOString(); st.pos++; show(); }
function back(){ if (st.pos>0){ st.pos--; document.getElementById("main").style.display="block"; document.getElementById("done").style.display="none"; show(); } }
function save(){
  const out = {manifest_sha256: __MHASH__, shown_order: st.queue.slice(0, st.pos), answers: st.answers, answered_at: st.served};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "g3_answers.json"; a.click();
}
document.getElementById("A").onclick = e => window.open(e.target.src);
document.getElementById("B").onclick = e => window.open(e.target.src);
document.addEventListener("keydown", e => { const k = e.key.toLowerCase();
  if (k==="s") ans("same"); else if (k==="d") ans("different"); else if (k==="c") ans("cant_tell"); });
show();
</script></body></html>
"""


def main(a):
    import hashlib
    d = Path(a.pairs)
    raw = (d / "manifest.json").read_bytes()
    m = json.loads(raw)
    groups = groups_from_manifest(m)
    order = [p["id"] for g in groups for p in g["sample"]]
    np.random.default_rng(SEED).shuffle(order)
    G = [dict(ids=[p["id"] for p in g["sample"] + g["reserve"]], reserve=[p["id"] for p in g["reserve"]])
         for g in groups]
    html = (HTML.replace("__GROUPS__", json.dumps(G)).replace("__ORDER__", json.dumps(order))
                .replace("__MHASH__", json.dumps(hashlib.sha256(raw).hexdigest())).replace("__MIN__", str(MIN_DECIDED)))
    (d / "g3_check.html").write_text(html)
    json.dump([dict(group=g["group"], bins=g["bins"], n_pairs_total=g["n_pairs_total"], n_sample=len(g["sample"]),
                    n_reserve=len(g["reserve"])) for g in groups], open(d / "g3_groups.json", "w"), indent=1)
    print(f"wrote {d / 'g3_check.html'}: {len(order)} sample pairs in {len(groups)} groups")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=".")
    main(ap.parse_args())
