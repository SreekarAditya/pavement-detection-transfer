"""Screen releases, commit decisions with calibration evidence, and score a sealed holdout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps

import audit_policy as audit
import pr_common as PC
import pr_exp18 as E18
from conformance import save_csv, write_json

HIGH_DOI="10.17632/sywswj7djj.1"
HIGH_RELEASE="https://data.mendeley.com/datasets/sywswj7djj/1"
HIGH_URL="https://data.mendeley.com/public-files/datasets/sywswj7djj/files/518cfb67-17f4-44a9-92a4-6d9d64cf8b83/file_downloaded"
HIGH_SIZE=1704448333
RTK_RELEASE="https://data.mendeley.com/datasets/hssswvmjwf/1"
RTK_DOC="https://github.com/thiagortk/Road-surface-detection-and-differentiation-considering-surface-damages"
WEIGHT_SHA="3fe3eaf8fa8f9300d11286859f229aa61f151a06f11d334d654f716d6a63d57d"
FORM=re.compile(r"(?P<parent>DJI_\d{14}_\d{4}|\d+x\d+)_640_(?P<row>\d+)_(?P<col>\d+)\.jpg",re.I)
POTHOLE=3
FLOOR=.25
K=15.80


def need(test, text):
    if not test:
        raise ValueError(text)


def clip_box(x,y,w,h,size):
    raw=np.array([x-w/2,y-h/2,x+w/2,y+h/2])*size
    return np.clip(raw,0,size)


def parent(name):
    match=FORM.fullmatch(name)
    return (match["parent"].lower(),int(match["row"]),int(match["col"])) if match else None


def load_highrpd(path, dest):
    images=dest/"images"
    images.mkdir(parents=True)
    with zipfile.ZipFile(path) as z:
        jpg={Path(n).name:n for n in z.namelist() if re.fullmatch(r"HighRPD/images/[^/]+\.jpg",n,re.I)}
        txt={Path(n).stem:n for n in z.namelist() if re.fullmatch(r"HighRPD/labels/[^/]+\.txt",n,re.I)}
        need(len(set(z.namelist()))==len(z.namelist()), "Duplicate ZIP member")
        need(jpg and set(Path(n).stem for n in jpg)==set(txt), "The original release does not pair every tile and label")
        names=sorted(jpg)
        source=[parent(n) for n in names]
        need(all(source) and len(set(source))==len(source),"Missing or duplicate tile-to-parent identifier")
        counts=Counter(s[0] for s in source)
        need(any(n>1 for n in counts.values()) and all(0<=p[1]<9 and 0<=p[2]<13 for p in source),
             "Tile indices are inconsistent with the declared parent geometry")
        records=[]
        clipped=degenerate=0
        for f in names:
            b=z.read(jpg[f])
            (images/f).write_bytes(b)
            with Image.open(images/f) as im:
                need(ImageOps.exif_transpose(im).size==(640,640),"Unexpected tile dimensions")
            for line in z.read(txt[Path(f).stem]).decode().splitlines():
                if not line.strip():
                    continue
                cls,*coords=line.split()
                need(cls in ("0","1","2") and len(coords)==4,"Unrecognized YOLO annotation")
                v=list(map(float,coords))
                need(np.isfinite(v).all() and min(v[2:])>0,"Invalid YOLO coordinates")
                if cls!="2":
                    continue
                x1,y1,x2,y2=clip_box(*v,640)
                clipped+=int(x1==0 or y1==0 or x2==640 or y2==640)
                if x2-x1<1 or y2-y1<1:
                    degenerate+=1
                    continue
                records.append((f,POTHOLE,x1,y1,x2,y2))
    labels=pd.DataFrame(records,columns=["file","cls",*PC.BOX]).drop_duplicates()
    return names,source,labels,dict(parents_in_release=len(counts),reported_source_parents=546,
                                    boxes_at_image_boundary=clipped,degenerate_pothole_boxes=degenerate)


def links(G,Q,R,qph,rph,cosine,same=False):
    # Preserve G4's cosine computation. A bounded vectorized Hamming pass avoids
    # allocating a 64-bit expansion of every cross-survey comparison.
    out=G.links(Q,R,np.array([],dtype=np.uint64),np.array([],dtype=np.uint64),cosine,G.LINK_PH,same=same)
    for start in range(0,len(qph),128):
        d=np.bitwise_count(qph[start:start+128,None]^rph[None,:])
        found=d<=G.LINK_PH
        if same:
            found &= np.arange(len(rph))[None,:] > np.arange(start,start+len(d))[:,None]
        i,j=np.nonzero(found)
        out.update(zip((i+start).tolist(),j.tolist()))
    return out


def reference_sets(a, G, C0, model):
    rph,rd,rnames,c0sha=E18.rdd_reference(a,a.c0,True)
    uph,ud,unames=E18.used_reference(a.root,True)
    earlier=SimpleNamespace(raw=a.sources/"prior",ext18=a.sources/"prior-record",candidates=a.rules/"exp18_candidates.csv")
    E18.download(earlier,{})
    ph=[C0.hex_to_u64(rph),C0.hex_to_u64(uph)]
    vectors=[rd,ud]; names=rnames+unames
    prior_sources=[]
    for candidate in pd.read_csv(earlier.candidates).itertuples():
        top=earlier.raw/candidate.slug/"files"
        paths=sorted(p for p in top.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"))
        need(paths, f"No reference images for {candidate.slug}")
        h=G.hash_table([str(p) for p in paths],[str(p.relative_to(top)) for p in paths])
        need(not h.phash.isna().any(),f"Unreadable reference images for {candidate.slug}")
        ph.append(C0.hex_to_u64(h.phash))
        vectors.append(C0.embed([str(p) for p in paths],model,"cpu",workers=a.workers))
        names.extend(f"{candidate.slug}:{p.relative_to(top)}" for p in paths)
        prior_sources.append(dict(release=candidate.slug,image_count=len(paths),archive_digests=candidate.sha256))
    return np.concatenate(ph),np.concatenate(vectors),names,dict(c0_file_hashes=c0sha,earlier_external=prior_sources)


def screen(a):
    import torch
    PC.locked_scripts(a.root)
    import pdt_s2_g4 as G
    import pdt_c0 as C0
    torch.set_num_threads(a.workers)
    G.NH=G.NW=a.workers
    C0.popcount64=np.bitwise_count
    params=PC.load_params(a.rules)
    a.out.mkdir(parents=True)
    a.sources.mkdir(parents=True,exist_ok=True)
    results=[]
    high=dict(release="HighRPD v1", doi=HIGH_DOI, url=HIGH_RELEASE, license="CC BY 4.0",
              team="He et al., Southwest Jiaotong University, Shanxi acquisition",
              original_labels="YOLO class 2 to D40", index=0,
              parent_parser=FORM.pattern, prior_inspection="Release metadata and ZIP directory; no study scoring")
    try:
        z=a.sources/"HighRPD.zip"
        if not z.is_file():
            stage=z.with_suffix(".partial")
            subprocess.run(["curl","-fLsS","--retry","2","--connect-timeout","20","-o",str(stage),HIGH_URL],check=True)
            stage.rename(z)
        need(z.stat().st_size==HIGH_SIZE,"Original HighRPD archive length differs from the published release")
        high["archive_sha256"]=PC.sha(z)
        dest=a.sources/"highrpd"
        names,source,L,details=load_highrpd(z,dest)
        high.update(details)
        high["parent_count_note"]="The complete filename-derived parent identifiers govern. The release description reports 546 selected source photos."
        c0meta=json.loads((a.c0/"c0_meta.json").read_text())
        model,weights=C0.load_sscd("cpu")
        need(model is not None and PC.sha(weights)==c0meta["descriptor_file_sha256"],"SSCD is unavailable or differs")
        rph,rd,rnames,provenance=reference_sets(a,G,C0,model)
        high.update(provenance)
        files=[str(dest/"images"/f) for f in names]
        h=G.hash_table(files,names)
        need(not h.phash.isna().any(),"Unreadable HighRPD tiles")
        ph=C0.hex_to_u64(h.phash)
        transformed, groups=E18.transformed_copies(files,names,ph,C0,G.LINK_PH)
        drop=set(transformed)
        filtered=[i for i,f in enumerate(names) if f not in drop]
        names=[names[i] for i in filtered]; source=[source[i] for i in filtered]
        files=[files[i] for i in filtered]; ph=ph[filtered]
        q=C0.embed(files,model,"cpu",workers=a.workers)
        matches=links(G,q,rd,ph,rph,G.LINK_COS)
        removed={i for i,_ in matches}
        match_rows=[dict(tile=names[i],linked_reference=rnames[j],reason="frozen perceptual hash or SSCD link")
                    for i,j in sorted(matches)]
        if match_rows:
            save_csv(a.out/"overlap_removals.csv",match_rows)
        ix=np.array([i for i in range(len(names)) if i not in removed],dtype=int)
        kept=[names[i] for i in ix]
        kp=[source[i][0] for i in ix]
        L=L[L.file.isin(set(kept))]
        layout=links(G,q[ix],q[ix],ph[ix],ph[ix],G.BLOCK_COS,same=True)
        first={}
        for i,p in enumerate(kp):
            j=first.setdefault(p,i)
            if i!=j: layout.add((j,i))
        blocks={f:f"highrpd_{cluster}" for f,cluster in zip(kept,G.components(len(kept),layout))}
        after=dict(positive_images=int(L.file.nunique()),mapped_boxes=len(L),blocks=len(set(blocks.values())))
        eligible=after["positive_images"]>=200 and after["mapped_boxes"]>=150 and after["blocks"]>=20
        high.update(transformed_removed=len(transformed),transformed_groups=groups,removed_overlap=len(removed),
                    images_after=len(kept),**after,status="eligible" if eligible else "ineligible",
                    reason="passed all release, independence and endpoint gates" if eligible else "positive images, mapped boxes or block gate failed")
        if eligible:
            split=params["split"]
            cal,held=PC.make_split(kept,blocks,split["calibration_share"],split["seed"],split["balance"])
            need(cal and held and set(cal).isdisjoint(held),"Empty or overlapping split")
            need({blocks[f] for f in cal}.isdisjoint({blocks[f] for f in held}),"A block crosses the split")
            o=a.out/"highrpd"
            o.mkdir()
            for name,keys in (("calibration",cal),("heldout",held)):
                (o/f"{name}.txt").write_text("\n".join(keys)+"\n")
                kind=o/("calibration_labels.parquet" if name=="calibration" else "private/heldout_labels.parquet")
                kind.parent.mkdir(exist_ok=True)
                L[L.file.isin(set(keys))].to_parquet(kind,index=False)
                high[f"{name}_labels_sha256"]=PC.sha(kind)
            save_csv(o/"blocks.csv",[dict(file=f,block=blocks[f],parent=p) for f,p in zip(kept,kp)])
            high["split_sha256"]={name:PC.sha(o/name) for name in ("calibration.txt","heldout.txt","blocks.csv")}
            high["image_source"]=str(dest/"images")
    except (ValueError,OSError,RuntimeError,SystemExit,subprocess.SubprocessError,zipfile.BadZipFile) as error:
        if str(error).startswith(("Missing or duplicate tile-to-parent","Tile indices","The original release does not pair")):
            high.update(status="ineligible",reason=str(error))
        else:
            high.update(status="pending",reason=f"Access or operational dependency: {error}")
    results.append(high)
    results.append(dict(release="RTK v1",index=1,doi="10.17632/hssswvmjwf.1",url=RTK_RELEASE,
                        author_documentation=RTK_DOC,license="CC BY 4.0",team="Rateke et al., Santa Catarina",
                        status="ineligible",reason="Official semantic annotation supplies class pixels. No per-instance identity or documented one-component-per-pothole mapping was established.",
                        prior_inspection="Release metadata and annotation documentation; no mask conversion or study scoring",
                        archive_sha256="Not fetched because the annotation-semantic gate failed"))
    write_json(a.out/"eligibility.json",results)
    save_csv(a.out/"eligibility.csv",[{k:r.get(k) for k in ("release","doi","status","reason","positive_images","mapped_boxes","blocks")} for r in results])
    (a.out/"REPORT.md").write_text("# Task 3 · Candidate eligibility\n\n"+"\n".join(
        f"- {x['release']} — {x['status']}. {x['reason']}" for x in results)+"\n")
    print("Both external candidates have documented screening dispositions.",flush=True)


def read_blocks(path):
    tab=pd.read_csv(path/"blocks.csv",dtype=str)
    return dict(zip(tab.file,tab.block))


def keys(path, name):
    items=(path/f"{name}.txt").read_text().splitlines()
    need(items and len(items)==len(set(items)),"Missing or duplicated split item")
    return items


def compute_commit(a):
    PC.locked_scripts(a.root)
    d=a.split
    need(not (d/"private").exists() and not (d/"heldout_labels.parquet").exists(),
         "The decision environment contains held-out labels")
    eligible=json.loads(a.eligibility.read_text())[0]
    need(eligible["status"]=="eligible","The frozen eligibility check did not pass")
    for name,expected in eligible["split_sha256"].items():
        need(PC.sha(d/name)==expected,"Split digest changed")
    for name,want in ((d/"calibration_labels.parquet",eligible["calibration_labels_sha256"]),):
        need(PC.sha(name)==want,"Calibration labels changed")
    cal=keys(d,"calibration"); B=read_blocks(d)
    cb=sorted({B[f] for f in cal}); bidx={b:i for i,b in enumerate(cb)}
    ci=np.array([bidx[B[f]] for f in cal])
    image_counts=np.bincount(ci,minlength=len(cb))
    P=pd.read_parquet(a.calibration)
    need(P.file.isin(set(cal)).all(),"Prediction contains an out-of-split item")
    N=yaml.safe_load((a.rules/"exp1_constants.yaml").read_text())["values"]["pothole"]["unseen"]
    p=(P[P.cls==POTHOLE])
    order=audit.ordered_blocks(len(cb),[2026,303,0,0])
    # Threshold sample uses unlabeled prediction confidences and all images in its selected blocks.
    limit=min(len(cb),np.searchsorted(np.cumsum(image_counts[order]),N["n"])+1)
    included=set(order[:limit])
    sample=[file for file,i in zip(cal,ci) if i in included]
    t=PC.threshold_for_rate(np.sort(p[p.file.isin(sample)].conf.to_numpy())[::-1],len(sample),K)
    need(np.isfinite(t),"Recalibration supplied no finite threshold")
    L=pd.read_parquet(d/"calibration_labels.parquet")
    T=PC.ThresholdRun(P,L,cal,[POTHOLE])
    _,m,x=T.per_image_at(t)
    mb=np.bincount(ci,weights=m,minlength=len(cb))
    xb=np.bincount(ci,weights=x,minlength=len(cb))
    o=audit.ordered_blocks(len(cb),[2026,303,0,1])
    arms, first, second=audit.policies(o,image_counts,mb,xb,FLOOR)
    count=min(len(cb),np.searchsorted(np.cumsum(xb[o]),N["M"])+1)
    select=o[:count]
    p,lo,hi,se=PC.ratio_ci(mb[select],xb[select],audit.Z90)
    arms["original_box_audit"]=dict(action="accept" if np.isfinite(lo) and lo>=FLOOR else "adapt",
        action_reason="frozen lower-bound action",images=int(image_counts[select].sum()),boxes=int(xb[select].sum()),
        blocks=int(count),target=int(N["M"]),recall=p,lo90=lo,hi90=hi,se=se,
        audit_shortfall=bool(xb[select].sum()<N["M"]))
    need(a.weight_sha==WEIGHT_SHA,"Locked checkpoint differs")
    return dict(candidate="HighRPD v1",doi=HIGH_DOI,index=0,code_commit=a.code_commit,
        threshold=t,threshold_images=len(sample),threshold_target=N["n"],threshold_shortfall=len(sample)<N["n"],
        selected_model="yolo12m_seed0", checkpoint_sha256=a.weight_sha, floor=FLOOR,alert_target=K,arms=arms,
        first_look=first,full_look=second,threshold_blocks=[cb[i] for i in order[:limit]],
        audit_block_order=[cb[i] for i in o],split_sha256=eligible["split_sha256"],eligibility_sha256=PC.sha(a.eligibility),
        calibration_prediction_sha256=PC.sha(a.calibration),heldout_prediction_sha256=PC.sha(a.heldout),
        calibration_labels_sha256=eligible["calibration_labels_sha256"],heldout_labels_sha256=eligible["heldout_labels_sha256"])


def commit(a):
    from datetime import datetime,timezone
    a.out.mkdir(parents=True)
    value=compute_commit(a)
    value["committed_at_utc"]=datetime.now(timezone.utc).isoformat()
    write_json(a.out/"commitment.json",value)


def score(a):
    PC.locked_scripts(a.root)
    signed=json.loads(a.commitment.read_text())
    need(PC.sha(a.commitment)==a.commitment_sha,"Commitment digest differs")
    git=subprocess.check_output(["git","-C",str(a.approved_repo),"show",f"{a.approved_tag}:{a.approved_path}"])
    need(git==a.commitment.read_bytes(),"A published or verified Git commitment is required before unsealing")
    # Recreate the commitment with no held-out reference labels in the decision input directory.
    a.code_commit=signed["code_commit"]; a.weight_sha=signed["checkpoint_sha256"]
    a.out.mkdir(parents=True)
    check=a.out/"recomputed.json"
    write_json(check,compute_commit(a))
    actual=json.loads(check.read_text())
    signed.pop("committed_at_utc")
    need(actual==signed,"The decision changed since the published commitment")
    need(PC.sha(a.labels)==actual["heldout_labels_sha256"],"Held-out label checksum differs")
    h=keys(a.split,"heldout")
    P=pd.read_parquet(a.heldout); L=pd.read_parquet(a.labels)
    need(P.file.isin(set(h)).all() and L.file.isin(set(h)).all(),"An input item is outside the held-out split")
    T=PC.ThresholdRun(P,L,h,[POTHOLE])
    al,m,n=T.per_image_at(signed["threshold"])
    truth=float(m.sum()/n.sum()) if n.sum() else None
    rate=float(100*al.sum()/len(h))
    B=read_blocks(a.split)
    names=sorted({B[f] for f in h}); idx={b:i for i,b in enumerate(names)}
    b=np.array([idx[B[f]] for f in h])
    data=[np.bincount(b,weights=x,minlength=len(names)) for x in (al,m,n)]
    draws=np.random.default_rng(np.random.SeedSequence([2026,305,0])).multinomial(len(names),np.full(len(names),1/len(names)),size=2000)
    alerts,matched,refs=[draws @ x for x in data]
    images=draws @ np.bincount(b,minlength=len(names))
    rr=np.divide(matched,refs,out=np.full(len(refs),np.nan),where=refs>0)
    rows=[]
    for arm,item in signed["arms"].items():
        est=item["recall"]
        e=est-truth if est is not None and truth is not None else None
        action=item["action"]
        outcome=("unscorable" if truth is None else "unresolved" if action=="unresolved" else
                 "correct_accept" if action=="accept" and truth>=FLOOR else "false_accept" if action=="accept" else
                 "unnecessary_adaptation" if truth>=FLOOR else "correct_adapt")
        rows.append(dict(candidate="HighRPD v1",arm=arm,action=action,outcome=outcome,
                         reason=item["action_reason"],labeled_images=item["images"],labeled_boxes=item["boxes"],
                         audit_blocks=item["blocks"],heldout_images=len(h),heldout_refs=int(n.sum()),
                         heldout_matched=int(m.sum()),heldout_alerts=int(al.sum()),heldout_recall=truth,
                         heldout_rate=rate,relative_rate_error=rate/K-1,within_rate=abs(rate/K-1)<=.10,
                         estimated_recall=est,recall_error=e,within_recall=abs(e)<=.05 if e is not None else None))
    save_csv(a.out/"external_results.csv",rows)
    save_csv(a.out/"heldout_block_counts.csv",[dict(block=name,images=int((b==i).sum()),alerts=int(data[0][i]),
                   matched=int(data[1][i]),refs=int(data[2][i])) for i,name in enumerate(names)])
    write_json(a.out/"heldout_uncertainty.json",
               dict(heldout_recall_ci95=[float(x) for x in PC.pct(rr)] if np.isfinite(rr).any() else None,
                    workload_ci95=[float(x) for x in PC.pct(100*alerts/images)],bootstrap_replicates=len(draws)))
    (a.out/"REPORT.md").write_text("# Task 3 · Committed HighRPD transfer case\n\n"
      "The detector evaluated tiled input images. Alert workload is measured per 100 tiles. "
      "The frozen decision was recomputed before the held-out labels were opened. "
      "See the results table for both decision errors and unresolved cases. "
      "This is one independently produced survey case.\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    for name,fn in (("screen",screen),("commit",commit),("score",score)):
        s=sub.add_parser(name);s.set_defaults(run=fn)
        for key in ("root","rules","out"): s.add_argument("--"+key,required=True,type=Path)
        if name=="screen":
            for key in ("c0","sources"): s.add_argument("--"+key,required=True,type=Path)
            s.add_argument("--workers",type=int,default=8)
        else:
            for key in ("split","eligibility","calibration","heldout"): s.add_argument("--"+key,required=True,type=Path)
        if name=="commit":
            for key in ("code_commit","weight_sha"): s.add_argument("--"+key,required=True)
        if name=="score":
            for key in ("commitment","labels","approved_repo"): s.add_argument("--"+key,required=True,type=Path)
            for key in ("commitment_sha","approved_tag","approved_path"): s.add_argument("--"+key,required=True)
    args=p.parse_args()
    args.run(args)
