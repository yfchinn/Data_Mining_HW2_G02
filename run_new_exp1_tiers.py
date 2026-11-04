"""Three tiers on new pre-split P1-P4 outer 60/20/20, seed 42."""
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import data_preparation_exp1_updated_v2 as prep
from src.models import ThreeSigmaDetector, IsolationForestDetector, LogisticRegressionDetector
from src.evaluation import evaluate_scores, select_f1_threshold


def run(csv_path=None, output_dir='outputs/new_exp1_tiers/seed_42', seed=42):
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    modes=[m for m in prep.CONFIGS if m.startswith(('P1-','P2-','P3-','P4-'))]
    records=[]; counts=[]
    protocol=dict(seed=seed,scope='outer 60/20/20; no inner CV; exploratory pre-split sampling',
        source='data_preparation_exp1_updated_v2.get_prepared_data',modes=modes,
        thresholds='T1 fixed 3; T2/T3 validation best F1, frozen before test',
        limitation='different mode test populations; IQR pre-split; cannot interpret as common-test benchmark',
        models={},status='running')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for mode in modes:
        print('Preparing',mode,flush=True)
        started=perf_counter()
        data=prep.get_prepared_data(mode=mode,fold_index=None,csv_path=csv_path,random_state=seed)
        preparation_seconds=perf_counter()-started
        directory=output/mode;directory.mkdir()
        np.savez_compressed(directory/'split_ids.npz',**{f'{p}_row_ids':data[f'{p}_row_ids'] for p in ('train','val','test')})
        for part in ('train','val','test'):
            y=data[f'y_{part}']
            counts.append(dict(mode=mode,partition=part,rows=len(y),normal=int(y.eq(0).sum()),fraud=int(y.sum())))
        protocol['models'][mode]={}
        for tier,model in [('T1',ThreeSigmaDetector()),('T2',IsolationForestDetector(random_state=seed)),
                           ('T3',LogisticRegressionDetector(random_state=seed))]:
            started=perf_counter(); model.fit(data['X_train'],data['y_train']); elapsed=perf_counter()-started
            if tier!='T3': assert model.n_training_rows_==int(data['y_train'].eq(0).sum())
            val_scores=model.score_samples(data['X_val'])
            if tier!='T1':model.threshold_=select_f1_threshold(data['y_val'],val_scores)
            entry=dict(parameters=model.get_params(),threshold=model.threshold_,
                fit_rows=model.n_training_rows_,fit_frauds=0 if tier!='T3' else int(data['y_train'].sum()),
                training_seconds=elapsed,preparation_seconds=preparation_seconds,
                preprocessing='provided training-only Time/Amount scaler; T1/T2 use only V features')
            protocol['models'][mode][tier]=entry
            (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
            joblib.dump(dict(model=model,scaler=data['scaler'],feature_columns=data['feature_columns'],
                metadata=entry),directory/f'{tier}.joblib')
            for part in ('train','val','test'):
                started=perf_counter()
                scores=val_scores if part=='val' else model.score_samples(data[f'X_{part}'])
                score_seconds=perf_counter()-started
                metrics=evaluate_scores(data[f'y_{part}'],scores,model.threshold_)
                metrics['accuracy']=(metrics['tn']+metrics['tp'])/len(scores)
                records.append(dict(mode=mode,tier=tier,partition=part,seed=seed,
                    threshold=model.threshold_,fit_rows=model.n_training_rows_,training_seconds=elapsed,
                    score_ms_per_sample=score_seconds*1000/len(scores) if part!='val' else np.nan,**metrics))
                pd.DataFrame(dict(row_id=data[f'{part}_row_ids'],y_true=data[f'y_{part}'],score=scores,
                    threshold=model.threshold_,y_pred=(scores>=model.threshold_).astype(int))).to_csv(directory/f'{tier}_{part}_predictions.csv',index=False)
            print(mode,tier,'test AP',round(records[-1]['average_precision'],4),'F1',round(records[-1]['f1'],4),flush=True)
        pd.DataFrame(records).to_csv(output/'metrics.csv',index=False)
        pd.DataFrame(counts).to_csv(output/'split_counts.csv',index=False)
    protocol['status']='complete'
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    return pd.DataFrame(records)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/new_exp1_tiers/seed_42'))
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    run(args.csv,args.output_dir,args.seed)
