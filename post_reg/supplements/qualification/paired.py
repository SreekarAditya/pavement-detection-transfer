"""Reconcile archived E1 and compare policies on identical calibration prefixes."""
import argparse
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

import audit_policy as A
import pr_common as PC
from pr_exp1 import Net
from pr_exp2 import block_samples


def check(a, b, message):
    try:
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,equal_nan=True)
    except AssertionError as exc:
        raise ValueError(message) from exc


def summarize(draws):
    rows=[]
    for (coll, cs, arm), x in draws.groupby(["collection","class_set","arm"], sort=False):
        known=x[x.truth.notna()]
        resolved=known[known.action!="unresolved"]
        pos=known.truth.astype(bool)
        counts=dict(accept_pass=int(((known.action=="accept") & pos).sum()),
                    false_accept=int(((known.action=="accept") & ~pos).sum()),
                    correct_adapt=int(((known.action=="adapt") & ~pos).sum()),
                    unnecessary_adapt=int(((known.action=="adapt") & pos).sum()),
                    unresolved_scorable=int((known.action=="unresolved").sum()))
        z=dict(collection=coll,class_set=cs,arm=arm,draws=len(x),scorable=len(known),resolved=len(resolved),
               unscorable=len(x)-len(known),unresolved=int((x.action=="unresolved").sum()),
               coverage=len(resolved)/len(known) if len(known) else np.nan, **counts)
        for key in ("false_accept","unnecessary_adapt"):
            z[key+"_per_scorable"]=counts[key]/len(known) if len(known) else np.nan
            z[key+"_per_resolved"]=counts[key]/len(resolved) if len(resolved) else np.nan
        for col in ("images","boxes","blocks"):
            z["mean_labeled_"+col]=x[col].mean()
            z["median_labeled_"+col]=x[col].median()
        z.update(second_audit_frequency=x.second_audit.mean(),second_request_frequency=x.second_requested.mean(),
                 shortfalls=int((x.images<x.target).sum()), no_distinct_second=int(x.no_distinct_second.sum()),
                 zero_references=int((x.boxes==0).sum()),zero_se=int((x.se==0).sum()),
                 mean_abs_alert_rate_error=x.alert_rate_error.abs().mean())
        rows.append(z)
    return pd.DataFrame(rows)


def paired_uncertainty(H, HX, threshold_index, action, floor, rng, draws, group):
    n=len(HX)
    blocks=rng.multinomial(n,np.full(n,1/n),size=draws)
    # one multiplicity row applies to every threshold and arm in its replicate
    num=blocks.astype(float) @ H.T
    den=blocks @ HX
    rate=np.divide(num,den[:,None],out=np.full_like(num,np.nan),where=den[:,None]>0)
    known=np.isfinite(rate[:,threshold_index])
    truth=rate[:,threshold_index]>=floor
    terms={}
    for name, acts in action.items():
        fa=(acts=="accept")[None,:] & ~truth & known
        ua=(acts=="adapt")[None,:] & truth & known
        unresolved=(acts=="unresolved")[None,:] & known
        denom=known.sum(axis=1)
        terms[name]={
            label: np.divide(mask.sum(axis=1),denom,out=np.full(len(denom),np.nan),where=denom>0)
            for label,mask in (("false_accept",fa),("unnecessary_adapt",ua),("unresolved",unresolved))
        }
    rows=[]
    for l,r in combinations(action,2):
        for label,lv in terms[l].items():
            delta=lv-terms[r][label]
            q=PC.pct(delta) if np.isfinite(delta).any() else (np.nan,np.nan)
            rows.append(dict(**group,arm_a=l,arm_b=r,metric=label,bootstrap_replicates=draws,
                             estimable_replicates=int(np.isfinite(delta).sum()), lower95=q[0],upper95=q[1]))
    return pd.DataFrame(rows)


def endpoint(R,rules,params,constants,D,c,cs):
    st=time.monotonic()
    det=params["selected"]
    N=Net(R,c,cs,det,rules,params)
    old=D[(D.collection==c) & (D.class_set==cs)].sort_values("draw").reset_index(drop=True)
    n_draw=params["sample_draws"]
    assert old.draw.tolist()==list(range(n_draw))
    v=constants["values"][cs][c]
    floor=PC.recall_floor(constants["recall_floor"],cs)
    cidx,csi=PC.EXT_ALL.index(c),list(PC.CLASS_SET_CLASSES).index(cs)
    seed=params["seed"]
    samples=block_samples(N.cal,N.B,v["n"],n_draw,seed,cidx)
    cb=sorted(set(N.cal_block)); hb=sorted({N.B[f] for f in N.held})
    ca={b:i for i,b in enumerate(cb)}; ha={b:i for i,b in enumerate(hb)}
    ci=np.array([ca[b] for b in N.cal_block]); hi=np.array([ha[N.B[f]] for f in N.held])
    images=np.bincount(ci,minlength=len(cb))
    refs=np.bincount(ci,weights=N.Tc.R.nref,minlength=len(cb)).astype(int)
    hx=np.bincount(hi,weights=N.Th.R.nref,minlength=len(hb)).astype(int)
    rng_e1=np.random.default_rng(np.random.SeedSequence([seed,1,cidx,csi,v["M"]]))
    zd=norm.ppf(.5+params["exp3"]["decision_level"]/2)
    cache, comparisons, audits, lookpairs = {},[],[],[]
    for d, idx in enumerate(samples):
        conf=np.concatenate([N.cal_conf[i] for i in idx]) if len(idx) else np.array([])
        threshold=PC.threshold_for_rate(np.sort(conf)[::-1],len(idx),N.k)
        archived=old.iloc[d]
        check(threshold,archived.threshold,f"{c}/{cs}/{d}: threshold differs")
        if threshold not in cache:
            _, cm, _=N.Tc.per_image_at(threshold)
            alerts, hm, _=N.Th.per_image_at(threshold)
            m=np.bincount(ci,weights=cm,minlength=len(cb)).astype(int)
            h=np.bincount(hi,weights=hm,minlength=len(hb)).astype(int)
            cache[threshold]=(m,h,100*alerts.sum()/len(N.held))
        m,h,rate=cache[threshold]
        held=h.sum()/hx.sum() if hx.sum() else np.nan
        check([held,rate],[archived.true_recall,archived.heldout_rate],f"{c}/{cs}/{d}: held-out measures differ")
        perm=rng_e1.permutation(len(cb))
        k=min(np.searchsorted(np.cumsum(refs[perm]),v["M"])+1,len(perm))
        block_ids=perm[:k]
        p,lo,high,_=PC.ratio_ci(m[block_ids],refs[block_ids],zd)
        check([p,lo,high],[archived.stated_recall,archived.stated_lo,archived.stated_hi],
              f"{c}/{cs}/{d}: original pilot differs")
        bound="accept" if np.isfinite(lo) and lo>=floor else "adapt"
        assert bound==archived.decision and int(refs[block_ids].sum())==archived.pilot_boxes
        point="accept" if np.isfinite(p) and p>=floor else "adapt"
        dims=dict(collection=c,class_set=cs,draw=d,threshold=threshold,heldout_recall=held,
                  truth=bool(held>=floor) if np.isfinite(held) else None,heldout_rate=rate,
                  alert_rate_error=rate/N.k-1, floor=floor)
        comparisons.append(dict(**dims,point=point,lower_bound=bound,original_audit_boxes=int(refs[block_ids].sum()),
                                original_audit_images=int(images[block_ids].sum()),original_audit_blocks=int(k),
                                point_estimate=p, lower90=lo, upper90=high))
        perm=A.ordered_blocks(len(cb),[2026,302,cidx,csi,d])
        policy, first, full=A.policies(perm,images,m,refs,floor)
        # The policy receives only calibration aggregates, even when held-out outcomes are present.
        audits.extend(dict(**dims,arm=arm,**values) for arm,values in policy.items())
        for level in (90,95):
            valid=np.isfinite(held) and first["valid"] and full["valid"]
            lookpairs.append(dict(collection=c,class_set=cs,draw=d,level=level,
                           first_estimable=bool(np.isfinite(held) and first["valid"]),
                           full_estimable=bool(np.isfinite(held) and full["valid"]),
                           first_includes=bool(first[f"lo{level}"]<=held<=first[f"hi{level}"]) if first["valid"] and np.isfinite(held) else None,
                           full_includes=bool(full[f"lo{level}"]<=held<=full[f"hi{level}"]) if full["valid"] and np.isfinite(held) else None,
                           simultaneous=bool(first[f"lo{level}"]<=held<=first[f"hi{level}"] and full[f"lo{level}"]<=held<=full[f"hi{level}"]) if valid else None))
    paired=pd.DataFrame(audits)
    summary=summarize(paired)
    H=np.stack([values[1] for values in cache.values()])
    lookup={t:i for i,t in enumerate(cache)}
    tidx=np.array([lookup[t] for t in old.threshold])
    actions={a: paired[paired.arm==a].action.to_numpy() for a in A.ARMS}
    group=dict(collection=c,class_set=cs)
    uncertainty=paired_uncertainty(H,hx,tidx,actions,floor,np.random.default_rng(
        np.random.SeedSequence([2026,304,cidx,csi])),params["bootstrap"]["draws"], group)
    at=summary.set_index("arm")
    for metric in ("false_accept","unnecessary_adapt","unresolved"):
        key=metric+"_per_scorable"
        if metric=="unresolved":
            at[key]=at.unresolved_scorable/at.scorable
        sel=uncertainty.metric==metric
        uncertainty.loc[sel,"difference"]=[at.loc[a,key]-at.loc[b,key] for a,b in
                                           zip(uncertainty.loc[sel,"arm_a"],uncertainty.loc[sel,"arm_b"])]
    for col in ("images","boxes","blocks"):
        key="mean_labeled_"+col
        uncertainty["paired_mean_"+col+"_difference"]=[at.loc[a,key]-at.loc[b,key] for a,b in zip(uncertainty.arm_a,uncertainty.arm_b)]
    print(f"Verified and profiled {c}/{cs} in {time.monotonic()-st:.1f}s",flush=True)
    return paired,summary,pd.DataFrame(comparisons),uncertainty,pd.DataFrame(lookpairs)


def main(args):
    PC.locked_scripts(args.lock)
    params=PC.load_params(args.rules)
    constants=yaml.safe_load((args.rules/"exp1_constants.yaml").read_text())
    detector=params["selected"]
    original=pd.read_parquet(args.draws)
    D=original[(original.arm=="procedure") & (original.detector==detector)]
    tasks=[(c,cs) for c in PC.EXT_ALL for cs in ("four_class","pothole")
           if params["class_sets"][cs]["networks"]=="all_seven" or c in params["class_sets"][cs]["networks"]]
    assert len(tasks)==10 and len(D)==10*params["sample_draws"] and params["bootstrap"]["draws"]==2000
    args.out.mkdir(parents=True)
    batches=[endpoint(PC.Root(args.lock),args.rules,params,constants,D,c,cs) for c,cs in tasks]
    paired, summary, reconciliation, boot, coverage=(pd.concat(v,ignore_index=True) for v in zip(*batches))
    paired.to_parquet(args.out/"paired_draws.parquet",index=False)
    summary.to_csv(args.out/"endpoint_summary.csv",index=False)
    reconciliation.to_csv(args.out/"original_e1_reconciliation.csv",index=False)
    boot.to_csv(args.out/"paired_uncertainty.csv",index=False)
    coverage.groupby(["collection","class_set","level"]).agg(
        first_n=("first_includes","count"),first_inclusion=("first_includes","mean"),
        full_n=("full_includes","count"),full_inclusion=("full_includes","mean"),
        joint_n=("simultaneous","count"),joint_inclusion=("simultaneous","mean")
    ).to_csv(args.out/"coverage_diagnostics.csv")
    cols=["collection","class_set","arm","false_accept","unnecessary_adapt","unresolved","unscorable",
          "mean_labeled_images","mean_labeled_boxes","mean_labeled_blocks","second_audit_frequency"]
    lines=["# Task 2 · Retrospective audit policy comparison","",
           "Archived thresholds, held-out recall and workload, and original E1 audit quantities and actions were reconciled for every endpoint and draw.",
           "The rows below give decision errors, unresolved cases and actual labeling effort together. False accepts and unnecessary adaptations count all scorable draws; the CSV also gives rates among resolved draws.",
           "Repeated draws describe calibration variability on each fixed network. The paired intervals use joint held-out block resampling with the calibration draws fixed. The inclusion diagnostics are empirical.", "",
           "| "+" | ".join(cols)+" |","| "+" | ".join("---" for _ in cols)+" |"]
    lines.extend("| "+" | ".join(f"{x:.3f}" if isinstance(x,float) else str(x) for x in row)+" |"
                 for row in summary[cols].itertuples(index=False,name=None))
    (args.out/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for col in ("lock","rules","draws","out"):
        p.add_argument("--"+col,type=Path,required=True)
    main(p.parse_args())
