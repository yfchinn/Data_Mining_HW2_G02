"""Evaluate frozen P0 artifacts; never fit models or select thresholds on test."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from src.evaluation import evaluate_scores


def ap_cache(y, scores):
    order=np.argsort(-scores,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(scores[order]))+1]
    return order,starts,np.asarray(y)[order]


def weighted_ap(cache, weights):
    order,starts,y=cache
    w=weights[order]
    positives=np.add.reduceat(w*y,starts)
    totals=np.add.reduceat(w,starts)
    tp=np.cumsum(positives); total=np.cumsum(totals)
    precision=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
    return float(np.sum(positives*precision)/tp[-1])


def bootstrap_gap(y, first, second, repeats=1000, seed=20261102):
    """Paired stratified row bootstrap of mean per-run AP; same weights for all runs.

    Preserves class counts, captures test-row uncertainty conditional on fitted
    models; correlated model/fold runs are never resampled as independent data.
    """
    caches=[[ap_cache(y,s) for s in group] for group in (first,second)]
    def gap(weights):
        return np.mean([weighted_ap(c,weights) for c in caches[0]])-np.mean([weighted_ap(c,weights) for c in caches[1]])
    rng=np.random.default_rng(seed)
    populations=[np.flatnonzero(y==label) for label in (0,1)]
    deltas=[]
    for i in range(repeats):
        sampled=np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in populations])
        weights=np.bincount(sampled,minlength=len(y))
        deltas.append(gap(weights))
        if (i+1)%100==0: print('bootstrap',i+1,'/',repeats,flush=True)
    low,high=np.quantile(deltas,[.025,.975])
    return dict(metric='mean_per_run_average_precision',comparison='IF minus LR',
        observed_gap=float(gap(np.ones(len(y)))),ci_low=float(low),ci_high=float(high),
        significant_at_05=bool(low>0 or high<0),bootstrap_repeats=repeats,bootstrap_seed=seed,
        method='paired class-stratified transaction bootstrap; percentile 95% CI',
        limitation='conditional on trained models and observed class counts; assumes independent transactions; no card IDs for clustered bootstrap')


def main():
    output=Path('outputs/final_test')
    output.mkdir(parents=True,exist_ok=False)
    rawpath=Path('data/creditcard.csv')
    with rawpath.open('rb') as source: checksum=hashlib.file_digest(source,'sha256').hexdigest()
    raw=pd.read_csv(rawpath)
    # Freeze before obtaining any test scores.
    freeze=dict(mode='P0',seeds=[42,43,44],split_seed=42,
        T1='V14 two-sided 3sigma; threshold=3',
        T2='normal-only IF; V14,V10,V12,V4,V11,V17; 200 trees; 256 samples/tree',
        T3='all-feature LR; C=1; balanced; no resampling (P0 chosen for AP)',
        thresholds='use saved validation-selected thresholds without changes',
        data_sha256=checksum,bootstrap_repeats=1000,status='frozen')
    (output/'frozen_protocol.json').write_text(json.dumps(freeze,indent=2))
    records=[]; comparisons=[]
    for exp,folds in [('exp1',5),('exp2',1)]:
        root=Path('outputs/P0')/exp
        manifest=json.loads((root/'run.json').read_text())
        assert manifest['sha256']==checksum and manifest['seeds']==[42,43,44]
        val_metrics=pd.read_csv(root/'metrics.csv')
        first=[]; second=[]; reference_ids=None
        for fold in range(folds):
            ids=np.load(root/f'fold_{fold}'/'split_ids.npz')['test_row_ids']
            if reference_ids is not None: np.testing.assert_array_equal(reference_ids,ids)
            reference_ids=ids
            truth=raw.iloc[ids]['Class'].to_numpy()
            directory=output/exp/f'fold_{fold}'; directory.mkdir(parents=True)
            for seed in (42,43,44):
                for tier in ('tier1_three_sigma','tier2_isolation_forest','tier3_logistic_regression'):
                    artifact=root/f'fold_{fold}'/f'{tier}_seed_{seed}.joblib'
                    bundle=joblib.load(artifact); model=bundle['model']
                    X=raw.iloc[ids][bundle['feature_columns']].astype(float).copy()
                    started=perf_counter()
                    X[['Time','Amount']]=bundle['scaler'].transform(X[['Time','Amount']])
                    preprocess_seconds=perf_counter()-started
                    started=perf_counter(); scores=model.score_samples(X); elapsed=perf_counter()-started
                    old=val_metrics.loc[(val_metrics.fold==fold)&(val_metrics.seed==seed)&(val_metrics.tier==tier)].iloc[0]
                    assert model.threshold_==old.threshold or np.isclose(model.threshold_,old.threshold,rtol=0,atol=1e-15)
                    record=dict(experiment=exp,fold=fold,seed=seed,tier=tier,partition='test',
                        threshold=model.threshold_,training_seconds=old.training_seconds,
                        score_ms_per_sample=elapsed*1000/len(ids),
                        preprocess_and_score_ms_per_sample=(preprocess_seconds+elapsed)*1000/len(ids),
                        evaluation_rows=len(ids),evaluation_frauds=int(truth.sum()),
                        **evaluate_scores(truth,scores,model.threshold_))
                    records.append(record)
                    pd.DataFrame(dict(row_id=ids,y_true=truth,score=scores,threshold=model.threshold_,
                        y_pred=(scores>=model.threshold_).astype(int))).to_csv(directory/f'{tier}_seed_{seed}.csv',index=False)
                    if tier=='tier2_isolation_forest': first.append(scores)
                    if tier=='tier3_logistic_regression': second.append(scores)
                    print(exp,fold,seed,tier,'AP',round(record['average_precision'],4),'F1',round(record['f1'],4),flush=True)
        # Identical LR seeds can be deduplicated: each fold still has equal weight.
        second=second[::3]
        comparisons.append(dict(experiment=exp,**bootstrap_gap(truth,first,second)))
    frame=pd.DataFrame(records)
    frame.to_csv('results/final_test_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv('results/final_test_significance.csv',index=False)
    metrics=['average_precision','roc_auc','f1','precision','recall','fpr','training_seconds','score_ms_per_sample','preprocess_and_score_ms_per_sample']
    summaries=[]
    for exp,g in frame.groupby('experiment'):
        if exp=='exp1': summary=g.groupby(['fold','tier'])[metrics].mean().groupby('tier').agg(['mean','std'])
        else: summary=g.groupby('tier')[metrics].agg(['mean','std'])
        summary.columns=['_'.join(c) for c in summary.columns]
        summary['experiment']=exp; summaries.append(summary.reset_index())
    pd.concat(summaries).to_csv('results/final_test_summary.csv',index=False)
    freeze['status']='complete'
    freeze['timing']='single batched inference; training times reused from saved training runs; excludes disk IO; latency is batch amortized'
    Path('results/final_test_protocol.json').write_text(json.dumps(freeze,indent=2))
    (output/'frozen_protocol.json').write_text(json.dumps(freeze,indent=2))


if __name__=='__main__': main()
