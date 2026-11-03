"""IF contamination sensitivity study; validation only, not the main normal-only fit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data_preparation import ROOT, _module, iter_prepared_folds
from src.evaluation import evaluate_scores, select_f1_threshold

FEATURES = ['V14', 'V10', 'V12', 'V4', 'V11', 'V17']


def run(experiment, csv_path=None, output_dir=None, n_splits=5):
    output = Path(output_dir or ROOT/'outputs'/'IF_contamination'/experiment)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    protocol = dict(experiment=experiment, features=FEATURES, seeds=[42,43,44],
        split_seed=42, n_estimators=200, max_samples=256,
        scaler='fit once per fold on original normal training only; reused in all C modes',
        normal_population='all original normal training rows, unchanged across modes',
        evaluation='validation only; threshold maximizes validation F1',
        contamination='C0-C4 are actual training fraud inclusion, not sklearn contamination parameter',
        limitation='sensitivity study deliberately fits polluted training data; main comparison stays normal-only',
        status='running')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for fold, base in iter_prepared_folds(experiment, csv_path=csv_path, n_splits=n_splits, fit_normal_only=True):
        for seed in (42,43,44):
            variants = _module(experiment).prepare_contamination_data(base, random_state=seed)
            baseline = None
            normal_ids = set(base['train_row_ids'][base['y_train'].eq(0)])
            previous_frauds = set()
            for mode, data in variants.items():
                y = data['y_train'].to_numpy()
                ids = data['train_row_ids']
                assert set(ids[y==0]) == normal_ids
                assert previous_frauds <= set(ids[y==1])
                previous_frauds = set(ids[y==1])
                for part in ('val','test'):
                    np.testing.assert_array_equal(data[f'{part}_row_ids'], base[f'{part}_row_ids'])
                    pd.testing.assert_frame_equal(data[f'X_{part}'],base[f'X_{part}'])
                # Stable order: unchanged normal block, followed by included frauds.
                order = np.lexsort((ids, y))
                X = data['X_train'].iloc[order][FEATURES]
                # Intentional: fit ALL selected rows. The main detector's normal-only
                # wrapper would remove injected fraud and invalidate this ablation.
                model = IsolationForest(n_estimators=200,max_samples=256,
                    contamination='auto',random_state=seed,n_jobs=1).fit(X)
                scores = -model.score_samples(base['X_val'][FEATURES])
                threshold = select_f1_threshold(base['y_val'],scores)
                metrics = evaluate_scores(base['y_val'],scores,threshold)
                if baseline is None: baseline = metrics
                rows.append(dict(experiment=experiment,fold=fold,seed=seed,mode=mode,
                    normal_count=int((y==0).sum()),fraud_count=int(y.sum()),
                    contamination_pct=float(y.mean()*100),threshold=threshold,
                    delta_ap=metrics['average_precision']-baseline['average_precision'],
                    delta_f1=metrics['f1']-baseline['f1'],**metrics))
                pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
                print(experiment,fold,seed,mode,'fraud=',int(y.sum()),
                      'AP=',round(metrics['average_precision'],4),flush=True)
    result = pd.DataFrame(rows)
    summary=result.groupby('mode')[['fraud_count','contamination_pct','average_precision','f1','delta_ap','delta_f1']].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(output/'summary.csv')
    protocol['status']='complete'
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['exp1','exp2'],required=True)
    args=parser.parse_args()
    run(args.experiment)
