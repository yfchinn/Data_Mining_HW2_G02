"""Compare P0-P6 for fixed logistic regression on validation only."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.data_preparation import ROOT, _module, _apply_mode, iter_prepared_folds
from src.evaluation import evaluate_scores, select_f1_threshold
from src.models import LogisticRegressionDetector


def run(experiment='exp2', csv_path=None, output_dir=None, n_splits=5, seed=42, smote_ratio=.01,
        split_seed=42):
    output = Path(output_dir or ROOT / 'outputs' / 'T3_sampling' / experiment)
    output.mkdir(parents=True, exist_ok=True)
    configs = _module('exp1').sampling_configurations(smote_ratio)
    records = []
    protocol = dict(experiment=experiment, modes=[c[0] for c in configs], seed=seed, split_seed=split_seed,
                    smote_ratio=smote_ratio, C=1, class_weight='balanced',
                    evaluation='validation only', status='running')
    (output/'run.json').write_text(json.dumps(protocol, indent=2))
    for fold, base in iter_prepared_folds(experiment, csv_path=csv_path, n_splits=n_splits, random_state=split_seed):
        directory = output / f'fold_{fold}'
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/'split_ids.npz', **{f'{p}_row_ids':base[f'{p}_row_ids'] for p in ('train','val','test')})
        for mode, _, _, _ in configs:
            print(f'{experiment} fold={fold} preparing {mode}', flush=True)
            started = perf_counter()
            data = _apply_mode(_module(experiment), base, mode, seed, smote_ratio)
            preprocessing_seconds = perf_counter()-started
            for part in ('val','test'):
                np.testing.assert_array_equal(data[f'{part}_row_ids'],base[f'{part}_row_ids'])
                pd.testing.assert_frame_equal(data[f'X_{part}'],base[f'X_{part}'])
            model = LogisticRegressionDetector(random_state=seed)
            started=perf_counter()
            model.fit(data['X_train'],data['y_train'])
            fit_seconds=perf_counter()-started
            scores=model.score_samples(data['X_val'])
            model.threshold_=select_f1_threshold(data['y_val'],scores)
            metrics=evaluate_scores(data['y_val'],scores,model.threshold_)
            records.append(dict(experiment=experiment,fold=fold,seed=seed,mode=mode,
                training_rows=len(data['y_train']),training_normal=int(data['y_train'].eq(0).sum()),
                training_fraud=int(data['y_train'].eq(1).sum()), threshold=model.threshold_,
                preprocessing_seconds=preprocessing_seconds,training_seconds=fit_seconds,**metrics))
            joblib.dump(dict(model=model,scaler=data['scaler'],feature_columns=data['feature_columns'],
                             metadata=data['metadata']),directory/f'{mode}.joblib')
            pd.DataFrame(dict(row_id=data['val_row_ids'],y_true=data['y_val'],score=scores,
                threshold=model.threshold_,y_pred=(scores>=model.threshold_).astype(int))).to_csv(directory/f'{mode}_val_predictions.csv',index=False)
            pd.DataFrame(records).to_csv(output/'metrics.csv',index=False)
            print(f'{mode}: AP={metrics["average_precision"]:.4f} F1={metrics["f1"]:.4f}',flush=True)
    frame=pd.DataFrame(records)
    summary=frame.groupby('mode')[['average_precision','roc_auc','f1','precision','recall','fpr']].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(output/'summary.csv')
    protocol['status']='complete'
    (output/'run.json').write_text(json.dumps(protocol,indent=2))
    return frame


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['exp1','exp2'],default='exp2')
    parser.add_argument('--csv',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--n-splits',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--split-seed',type=int,default=42)
    parser.add_argument('--smote-ratio',type=float,default=.01)
    args=parser.parse_args()
    run(args.experiment,args.csv,args.output_dir,args.n_splits,args.seed,args.smote_ratio,args.split_seed)
