"""T2 C0-C4 on a fixed new P1-10 outer split, with three model/subset seeds."""
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import data_preparation_exp1_updated_v2 as prep
from run_if_contamination import FEATURES
from src.evaluation import evaluate_scores, select_f1_threshold


def run(csv_path=None, output_dir='outputs/new_exp1_if_contamination', seeds=(42,43,44)):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    base = prep.get_prepared_data(mode='P1-10', fold_index=None,
                                  csv_path=csv_path, random_state=42)
    protocol = dict(source='data_preparation_exp1_updated_v2', mode='P1-10',
        split_seed=42, seeds=list(seeds), features=FEATURES,
        n_estimators=200, max_samples=256, sklearn_contamination='auto',
        fraud_counts='floor(min(360, available training fraud) * [0,1/12,1/4,1/2,1])',
        fit='all selected normal and fraud rows; normal population fixed',
        order='normal then fraud, each sorted by original row ID',
        threshold='validation best F1 per model, frozen for train/test',
        preprocessing='reuse P1-10 training-only scaler; selected V features unchanged',
        limitation='exploratory pre-split undersampling; test prevalence differs from original benchmark',
        status='running')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    np.savez_compressed(output/'base_split_ids.npz', **{
        f'{p}_row_ids':base[f'{p}_row_ids'] for p in ('train','val','test')})
    counts = [dict(partition=p, normal=int(base[f'y_{p}'].eq(0).sum()),
                   fraud=int(base[f'y_{p}'].eq(1).sum())) for p in ('train','val','test')]
    pd.DataFrame(counts).to_csv(output/'base_split_counts.csv',index=False)
    records=[]
    normal_ids=set(base['train_row_ids'][base['y_train'].eq(0)])
    for seed in seeds:
        previous=set()
        variants=prep.prepare_contamination_data(base,random_state=seed)
        baseline={}
        for mode,data in variants.items():
            y=data['y_train'].to_numpy()
            ids=data['train_row_ids']
            assert set(ids[y==0])==normal_ids
            selected_fraud=set(ids[y==1])
            assert previous <= selected_fraud
            previous=selected_fraud
            for part in ('val','test'):
                np.testing.assert_array_equal(data[f'{part}_row_ids'],base[f'{part}_row_ids'])
                pd.testing.assert_frame_equal(data[f'X_{part}'],base[f'X_{part}'])
            order=np.lexsort((ids,y))
            started=perf_counter()
            model=IsolationForest(n_estimators=200,max_samples=256,contamination='auto',
                                  random_state=seed,n_jobs=1).fit(data['X_train'].iloc[order][FEATURES])
            training_seconds=perf_counter()-started
            val_scores=-model.score_samples(base['X_val'][FEATURES])
            threshold=select_f1_threshold(base['y_val'],val_scores)
            directory=output/f'seed_{seed}'/mode
            directory.mkdir(parents=True)
            np.save(directory/'fit_row_ids.npy',ids[order])
            joblib.dump(dict(model=model,threshold=threshold,features=FEATURES,
                             scaler=base['scaler']),directory/'T2.joblib')
            for part in ('train','val','test'):
                # Score the unchanged full P1-10 training set for comparable train metrics.
                started=perf_counter()
                scores=val_scores if part=='val' else -model.score_samples(base[f'X_{part}'][FEATURES])
                elapsed=perf_counter()-started
                metrics=evaluate_scores(base[f'y_{part}'],scores,threshold)
                if mode=='C0': baseline[part]=metrics
                records.append(dict(seed=seed,mode=mode,partition=part,
                    normal_count=int((y==0).sum()),fraud_count=int(y.sum()),
                    contamination_pct=float(y.mean()*100),threshold=threshold,
                    training_seconds=training_seconds,
                    score_ms_per_sample=elapsed*1000/len(scores) if part!='val' else np.nan,
                    delta_ap=metrics['average_precision']-baseline[part]['average_precision'],
                    delta_f1=metrics['f1']-baseline[part]['f1'],**metrics))
                pd.DataFrame(dict(row_id=base[f'{part}_row_ids'],
                    y_true=base[f'y_{part}'].to_numpy(),score=scores,
                    y_pred=(scores>=threshold).astype(int))).to_csv(directory/f'{part}_predictions.csv',index=False)
            pd.DataFrame(records).to_csv(output/'metrics.csv',index=False)
            print(seed,mode,'fraud',int(y.sum()),'test AP',round(records[-1]['average_precision'],4),flush=True)
    result=pd.DataFrame(records)
    summary=result.groupby(['mode','partition'])[['contamination_pct','average_precision','f1',
        'precision','recall','fpr','delta_ap','delta_f1','training_seconds']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(output/'summary.csv')
    protocol['status']='complete'
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    return result


if __name__=='__main__':
    run()
