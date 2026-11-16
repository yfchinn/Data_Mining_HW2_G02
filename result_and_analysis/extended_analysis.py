#!/usr/bin/env python3
"""Reproducible follow-up analysis for NYCU Data Mining HW2 §§7–8.

This script intentionally uses only sklearn/numpy/pandas/scipy and reconstructs
P0 split definitions in the repo. It DOES NOT import scikit-learn-extra, which
may be unavailable in current Google Colab Python builds.

Outputs are retrospective analyses of the already examined ULB test set.
Threshold sweeps and contamination are validation-only and MUST NOT be used to
claim a fresh, untouched generalization test. No results are hard-coded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve)

FEATURES = ["Time", *[f"V{i}" for i in range(1,29)], "Amount"]
V_FEATURES = [f"V{i}" for i in range(1,29)]
FIXED6 = ["V14", "V10", "V12", "V4", "V11", "V17"]
CASES = [164455, 13796, 57615]
EXPECTED_HASH = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def check_dataset(path: Path, allow_small: bool=False):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. In Colab run git lfs pull or upload creditcard.csv.")
    with path.open("rb") as fh:
        prefix = fh.read(100)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError("creditcard.csv is a Git LFS POINTER, not the actual dataset. Run `git lfs pull`.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    required = FEATURES + ["Class"]
    if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
        raise ValueError("Missing Time/V1-V28/Amount/Class or non-finite input values")
    if not frame["Class"].isin([0,1]).all():
        raise ValueError("Unexpected class labels")
    if not allow_small:
        if len(frame)!=284807 or int(frame.Class.sum())!=492 or digest!=EXPECTED_HASH:
            raise ValueError("Not the frozen ULB CSV used in the report; check CSV hash/order/labels")
    frame = frame.reset_index(drop=True).copy()
    frame["row_id"] = np.arange(len(frame),dtype=int)
    return frame, digest


def get_folds(data: pd.DataFrame, experiment: str, n_folds: int=5):
    """Match data_preparation(exp1).py and data_preparation(exp2).py P0 raw indices."""
    if experiment=="exp1":
        dev, test = train_test_split(data, test_size=0.2, stratify=data.Class,
                                     random_state=42, shuffle=True)
        dev=dev.reset_index(drop=True); test=test.reset_index(drop=True)
        cv=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)
        for f,(t,v) in enumerate(cv.split(dev[FEATURES],dev.Class)):
            yield f,dev.iloc[t].copy(),dev.iloc[v].copy(),test.copy()
    elif experiment=="exp2":
        ordered=data.sort_values("Time",kind="mergesort")
        a=int(len(data)*0.6); b=int(len(data)*0.8)
        yield 0,ordered.iloc[:a].copy(),ordered.iloc[a:b].copy(),ordered.iloc[b:].copy()
    else: raise ValueError(experiment)


def ks_select(train: pd.DataFrame, k=6):
    """Use ONLY the current training partition's labels, never its validation/test."""
    normals=train.loc[train.Class.eq(0)]; frauds=train.loc[train.Class.eq(1)]
    scores={col:float(ks_2samp(normals[col],frauds[col]).statistic) for col in V_FEATURES}
    return sorted(V_FEATURES,key=lambda x:(-scores[x],int(x[1:])))[:k],scores


def select_f1(y, score):
    p,r,thresholds=precision_recall_curve(np.asarray(y),np.asarray(score))
    denom=p[:-1]+r[:-1]
    f1=np.divide(2*p[:-1]*r[:-1],denom,out=np.zeros_like(denom),where=denom>0)
    return float(thresholds[np.flatnonzero(f1==f1.max())[-1]])


def select_capacity_threshold(score, rate=.001):
    """Cap number of validation alerts at floor(rate*validation rows), with tie safety."""
    scores=np.asarray(score,float)
    k=max(1,int(np.floor(len(scores)*rate)))
    ordered=np.sort(scores)[::-1]
    threshold=float(ordered[k-1])
    if int((scores>=threshold).sum())>k:
        threshold=float(np.nextafter(threshold,np.inf))
    assert (scores>=threshold).sum()<=k
    return threshold,k


def stats(y,score,threshold):
    y=np.asarray(y,dtype=int); score=np.asarray(score,float)
    p=(score>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return dict(AP=float(average_precision_score(y,score)),ROC_AUC=float(roc_auc_score(y,score)),
                F1=float(f1),precision=float(precision),recall=float(recall),
                FPR=float(fp/(fp+tn)) if fp+tn else 0.,alerts=int(tp+fp),
                FP=int(fp),FN=int(fn),TP=int(tp),TN=int(tn))


def IF_fit(train,features,seed,n_trees=200,max_samples=256):
    return IsolationForest(n_estimators=n_trees,max_samples=max_samples,
                           contamination="auto",random_state=seed,n_jobs=1).fit(train[features])


def score_if(model,frame,features):
    return -model.score_samples(frame[features])


def score_t1(train,frame):
    # normal-only V14, ddof 0; score larger means more anomalous
    v=train.loc[train.Class.eq(0),"V14"].to_numpy()
    return np.abs((frame.V14.to_numpy()-v.mean())/v.std(ddof=0))


def mean_weighted_ap_cache(y,s):
    """Exact weighted sklearn average_precision_score with tied score groups."""
    order=np.argsort(-s,kind="stable")
    start=np.r_[0,np.flatnonzero(np.diff(np.asarray(s)[order]))+1]
    return order,start,np.asarray(y)[order]


def weighted_AP(cache,weights):
    order,start,y=cache
    w=weights[order]
    hits=np.add.reduceat(w*y,start)
    totals=np.add.reduceat(w,start)
    cumhits=np.cumsum(hits); cumtot=np.cumsum(totals)
    p=np.divide(cumhits,cumtot,out=np.zeros(len(cumtot),float),where=cumtot!=0)
    return float(np.dot(hits,p)/cumhits[-1])


def paired_bootstrap(y,first,second,repeats=1000,seed=20261102):
    """Paired class-stratified transaction bootstrap on MEAN AP of trained runs.

    Resamples the SAME transaction weights for both methods/all runs. Does not
    treat folds or seeds as independent test populations.
    """
    y=np.asarray(y,int)
    a=[mean_weighted_ap_cache(y,np.asarray(s)) for s in first]
    b=[mean_weighted_ap_cache(y,np.asarray(s)) for s in second]
    def gap(w):return np.mean([weighted_AP(c,w) for c in a])-np.mean([weighted_AP(c,w) for c in b])
    observed=gap(np.ones(len(y),dtype=int))
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(y==label) for label in (0,1)]
    replicates=np.empty(repeats)
    for i in range(repeats):
        drawn=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        weights=np.bincount(drawn,minlength=len(y))
        replicates[i]=gap(weights)
    low,high=np.quantile(replicates,[.025,.975])
    return dict(mean_AP_gap=float(observed),CI_low=float(low),CI_high=float(high),
                excludes_zero=bool(low>0 or high<0),repeats=repeats,
                unit="paired test transaction, class-stratified",\
                caution="conditional on observed test set and trained models; card dependence unknown")


def path_usage(model,one,features):
    """Frequency of each feature encountered on trees' DECISION paths.

    This is diagnostic split usage, not a signed additive SHAP contribution.
    """
    x=one[features].to_numpy(dtype=float)
    counts=Counter()
    for tree,sub in zip(model.estimators_,model.estimators_features_):
        path=tree.decision_path(x[:,sub]).indices
        for node in path:
            local=int(tree.tree_.feature[node])
            if local>=0: counts[features[int(sub[local])]]+=1
    return counts


def persist(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",default="data/creditcard.csv")
    p.add_argument("--out",default="outputs/extended_analysis")
    p.add_argument("--experiments",nargs="+",choices=["exp1","exp2"],default=["exp1","exp2"])
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--exp1-folds",type=int,default=5)
    p.add_argument("--bootstrap-repeats",type=int,default=1000)
    p.add_argument("--capacity-rate",type=float,default=.001)
    p.add_argument("--contam-total",type=int,default=0,
                   help="0 means use ALL original normal training rows as fixed total (recommended); smaller positive number for pilot")
    p.add_argument("--contam-max",type=int,default=360)
    p.add_argument("--skip-sweep",action="store_true")
    p.add_argument("--allow-small-test-data",action="store_true",help="Smoke-test ONLY; never cite as ULB")
    args=p.parse_args()
    if not (0<args.capacity_rate<1):raise ValueError("capacity-rate must be (0,1)")
    if args.contam_total<0:raise ValueError("contam-total cannot be negative")
    raw,digest=check_dataset(Path(args.csv),args.allow_small_test_data)
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    records=[]; selected_rows=[]; budget_rows=[]; threshold_rows=[]; contam_rows=[]; attr_rows=[]; sweep_rows=[]; bootstrap_rows=[]; bootstrap_data={}
    protocol=dict(dataset_sha256=digest,data_rows=len(raw),feature_sets="fixed six versus train-fold-only KS six",
                  threshold="F1 from validation; capacity rate set on validation, separately",
                  contamination="fixed count; replace normals with nested fraud subsets; validation only",
                  test_use="retrospective diagnostic on previously inspected P0 test, not independent fresh test",
                  seeds=args.seeds,exp1_folds=args.exp1_folds,
                  sklearn_IF=dict(n_estimators=200,max_samples=256,contamination="auto"))
    (root/"protocol.json").write_text(json.dumps(protocol,indent=2))
    for exp in args.experiments:
        for fold,train,val,test in get_folds(raw,exp,args.exp1_folds):
            nested,ks=ks_select(train)
            selected_rows.extend(dict(experiment=exp,fold=fold,feature=f,rank=i+1,KS=ks[f])
                                 for i,f in enumerate(nested))
            normal=train.loc[train.Class.eq(0)]
            # The T1 reference is deterministic across seeds. Recompute only once.
            t1_test=score_t1(train,test)
            t1_val=score_t1(train,val)
            if fold==0:
                for seed in args.seeds:
                    pass
            fixed_scores=[]; ref_scores=[]; test_truth=test.Class.to_numpy()
            for seed in args.seeds:
                fixed=IF_fit(normal,FIXED6,seed)
                nested_model=IF_fit(normal,nested,seed)
                v14_only=IF_fit(normal,["V14"],seed)
                sv=score_if(fixed,val,FIXED6); st=score_if(fixed,test,FIXED6)
                nv=score_if(nested_model,val,nested); nt=score_if(nested_model,test,nested)
                ov=score_if(v14_only,val,["V14"])
                tau=select_f1(val.Class,sv); nt_tau=select_f1(val.Class,nv)
                fixed_scores.append(st);ref_scores.append(t1_test)
                records.append(dict(experiment=exp,fold=fold,seed=seed,
                    variant="IF_V14_only",split="validation",features="V14",
                    threshold=select_f1(val.Class,ov),
                    **stats(val.Class,ov,select_f1(val.Class,ov))))
                for name,v_scores,t_scores,threshold,features in (
                    ("T1_3sigma",t1_val,t1_test,3.,["V14"]),
                    ("IF_fixed6",sv,st,tau,FIXED6),
                    ("IF_nestedKS6",nv,nt,nt_tau,nested),
                ):
                    for split,label,values in (("validation",val.Class,v_scores),("retrospective_test",test.Class,t_scores)):
                        records.append(dict(experiment=exp,fold=fold,seed=seed,variant=name,
                                            split=split,features=",".join(features),threshold=threshold,
                                            **stats(label,values,threshold)))
                # Capacity experiment: choose thresholds ONLY on validation.
                cap_tau,k=select_capacity_threshold(sv,args.capacity_rate)
                for name,threshold in (("validation_best_F1",tau),("validation_capacity",cap_tau)):
                    budget_rows.append(dict(experiment=exp,fold=fold,seed=seed,strategy=name,
                        assumed_alert_rate=args.capacity_rate,validation_alert_cap=k,
                        split="validation_only",threshold=threshold,
                        **stats(val.Class,sv,threshold)))
                # Validation-only local threshold sensitivity (never used to tune test).
                for shift in (-.05,-.025,0.,.025,.05):
                    candidate=tau+shift
                    threshold_rows.append(dict(experiment=exp,fold=fold,seed=seed,
                        delta_tau=shift,threshold=candidate,split="validation_only",
                        **stats(val.Class,sv,candidate)))
                # Score explanations: baseline vs median-replaced feature; path counts.
                if exp=="exp1":
                    for rowid in CASES:
                        example=test.loc[test.row_id.eq(rowid)]
                        if len(example)!=1:
                            if not args.allow_small_test_data: raise RuntimeError(f"Expected test row {rowid}")
                            continue
                        original=float(score_if(fixed,example,FIXED6)[0])
                        paths=path_usage(fixed,example,FIXED6)
                        for feature in FIXED6:
                            changed=example.copy()
                            median=float(normal[feature].median())
                            changed.loc[:,feature]=median
                            replaced=float(score_if(fixed,changed,FIXED6)[0])
                            attr_rows.append(dict(experiment=exp,fold=fold,seed=seed,row_id=rowid,
                                true_label=int(example.Class.iloc[0]),feature=feature,
                                observed=float(example[feature].iloc[0]),normal_train_median=median,
                                original_anomaly_score=original,median_replaced_score=replaced,
                                score_reduction=original-replaced,
                                decision_path_split_count=int(paths[feature]),
                                score_threshold=tau,
                                note="post-hoc single-feature perturbation; correlated PCA features and off-manifold values limit causal meaning"))
                # Single-variable controlled contamination experiment, validation ONLY.
                # Independent RNG per seed/fold; same candidate normal list across levels.
                rand=np.random.default_rng(20261116+seed+100*fold+(0 if exp=="exp1" else 1000))
                fraud=train.loc[train.Class.eq(1)]
                total=len(normal) if args.contam_total==0 else args.contam_total
                if len(normal)<total or len(fraud)==0 or total<=1: raise ValueError("Not enough training rows for requested contamination size")
                normals_idx=normal.sort_values("row_id").index.to_numpy()
                if args.contam_total: normals_idx=rand.choice(normals_idx,total,replace=False)
                fmax=min(len(fraud),args.contam_max,total-1)
                fraud_idx=rand.choice(fraud.index.to_numpy(),fmax,replace=False)
                count_levels=np.unique(np.floor(np.array([0,1/12,1/4,1/2,1])*fmax).astype(int))
                baseline_ap=None
                for n_fraud in count_levels:
                    # Fix TOTAL at contam_total: every injected fraud displaces one normal.
                    subset=pd.concat([normal.loc[normals_idx[:total-n_fraud]],
                                      fraud.loc[fraud_idx[:n_fraud]].sort_values("row_id")])
                    assert len(subset)==total
                    cm=IF_fit(subset,FIXED6,seed)
                    cs=score_if(cm,val,FIXED6)
                    ctau=select_f1(val.Class,cs)
                    metrics=stats(val.Class,cs,ctau)
                    if baseline_ap is None:baseline_ap=metrics["AP"]
                    contam_rows.append(dict(experiment=exp,fold=fold,seed=seed,
                        n_total=total,n_normal=total-n_fraud,n_fraud=n_fraud,
                        fraud_prevalence=float(n_fraud/total),
                        delta_AP_vs_size_matched_C0=metrics["AP"]-baseline_ap,
                        **metrics))
                # Small validation-only hyperparameter sweep. Fold 0 only to limit cost.
                if fold==0 and not args.skip_sweep:
                    for ntree,msamp in ((100,256),(200,256),(400,256),(200,512)):
                        sm=IF_fit(normal,FIXED6,seed,n_trees=ntree,max_samples=msamp)
                        scores=score_if(sm,val,FIXED6)
                        t=select_f1(val.Class,scores)
                        sweep_rows.append(dict(experiment=exp,fold=fold,seed=seed,
                            n_estimators=ntree,max_samples=msamp,
                            threshold=t,**stats(val.Class,scores,t)))
                print(f"{exp} fold={fold} seed={seed}  fixed_AP={stats(val.Class,sv,tau)['AP']:.4f}"
                      f" nested_AP={stats(val.Class,nv,nt_tau)['AP']:.4f}",flush=True)
            # Bootstrap original fixed IF vs T1 on SAME rows; repeated models not separate test populations.
            # Run once per protocol after all folds are aggregated, below.
            
            if exp not in bootstrap_data:bootstrap_data[exp]={"ids":test.row_id.to_numpy(),"y":test_truth,"if":[],"t1":[]}
            else:
                np.testing.assert_array_equal(bootstrap_data[exp]["ids"],test.row_id.to_numpy())
            bootstrap_data[exp]["if"].extend(fixed_scores)
            bootstrap_data[exp]["t1"].extend(ref_scores)
            persist(records,root/"comparisons.csv")
            persist(contam_rows,root/"size_matched_contamination_validation.csv")
    for exp,o in bootstrap_data.items():
        ci=paired_bootstrap(o["y"],o["if"],o["t1"],repeats=args.bootstrap_repeats)
        bootstrap_rows.append(dict(experiment=exp,comparison="IF_fixed6_minus_T1_3sigma",**ci))
    for name,rows in (
        ("comparisons.csv",records),("fold_selected_features.csv",selected_rows),
        ("capacity_threshold_validation.csv",budget_rows),
        ("threshold_sensitivity_validation.csv",threshold_rows),
        ("size_matched_contamination_validation.csv",contam_rows),
        ("case_feature_perturbation.csv",attr_rows),
        ("hyperparameter_sweep_validation.csv",sweep_rows),
        ("IF_vs_T1_paired_bootstrap.csv",bootstrap_rows),
    ):
        if rows:persist(rows,root/name)
    if attr_rows:
        mean=(pd.DataFrame(attr_rows).groupby(["row_id","feature"],as_index=False)
              .agg(mean_score_reduction=("score_reduction","mean"),
                   sd_score_reduction=("score_reduction","std"),
                   mean_path_split_count=("decision_path_split_count","mean")))
        mean.to_csv(root/"case_attribution_summary.csv",index=False)
    if contam_rows:
        (pd.DataFrame(contam_rows).groupby(["experiment","n_fraud"],as_index=False)
         .agg(n_runs=("AP","size"),mean_AP=("AP","mean"),sd_AP=("AP","std"),
              mean_delta_AP=("delta_AP_vs_size_matched_C0","mean"),
              sd_delta_AP=("delta_AP_vs_size_matched_C0","std"))
         .to_csv(root/"size_matched_contamination_summary.csv",index=False))
    print(f"All outputs in {root.resolve()}")

if __name__=="__main__": main()
