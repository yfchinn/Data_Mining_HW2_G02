"""Validation-only controlled IF diagnostics; all fits use normal P0 rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.data_preparation import get_prepared_data
from src.evaluation import evaluate_scores, select_f1_threshold


def main():
    out = Path('outputs/if_diagnostics')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    # All variants differ from baseline in one factor. No contamination or test scoring.
    configurations = [('baseline', None, 256), ('no_time', 'no_time', 256),
                      ('v_only', 'v_only', 256), ('v14_only', 'v14_only', 256),
                      ('eda_six', 'eda_six', 256), ('samples_1024', None, 1024),
                      ('samples_4096', None, 4096)]
    for experiment in ('exp2', 'exp1'):
        data = get_prepared_data(experiment, mode='P0', fold_index=0, fit_normal_only=True)
        normal = data['X_train'].loc[data['y_train'].eq(0)]
        val, y = data['X_val'], data['y_val']
        for name, subset, samples in configurations:
            columns = list(normal.columns)
            if subset == 'no_time': columns.remove('Time')
            if subset == 'v_only': columns = [f'V{i}' for i in range(1,29)]
            if subset == 'v14_only': columns = ['V14']
            if subset == 'eda_six': columns = ['V14','V10','V12','V4','V11','V17']
            for seed in (42,43,44):
                model = IsolationForest(n_estimators=200, max_samples=samples,
                                        random_state=seed, n_jobs=1).fit(normal[columns])
                scores = -model.score_samples(val[columns])
                threshold = select_f1_threshold(y, scores)
                row = dict(experiment=experiment, fold=0, variant=name, seed=seed,
                           max_samples=samples, threshold=threshold,
                           top100_frauds=int(y.iloc[np.argsort(-scores)[:100]].sum()),
                           **evaluate_scores(y, scores, threshold))
                records.append(row)
                pd.DataFrame(records).to_csv(out/'metrics.csv', index=False)
                print(experiment, name, seed, round(row['average_precision'],4), flush=True)
        # Inspect actual high-score normal examples against normal-training quantiles.
        model = IsolationForest(n_estimators=200,max_samples=256,random_state=42).fit(normal)
        scores = -model.score_samples(val)
        top = np.argsort(-scores)[:100]
        evidence = []
        for col in normal.columns:
            low, high = normal[col].quantile([.01,.99])
            top_values = val.iloc[top][col]
            evidence.append(dict(feature=col, train_p01=low, train_p99=high,
                                 top100_outside_fraction=float(((top_values<low)|(top_values>high)).mean()),
                                 val_normal_outside_fraction=float(((val.loc[y.eq(0),col]<low)|(val.loc[y.eq(0),col]>high)).mean())))
        pd.DataFrame(evidence).to_csv(out/f'{experiment}_tail_features.csv', index=False)
    frame = pd.DataFrame(records)
    summary = frame.groupby(['experiment','variant'])[['average_precision','f1','top100_frauds']].agg(['mean','std'])
    summary.to_csv(out/'summary.csv')
    (out/'protocol.json').write_text(json.dumps(dict(seeds=[42,43,44],fold=0,
        training='normal P0 only',evaluation='validation only',trees=200,
        feature_selection='V14 and six features fixed from team EDA; no KS recomputation',
        limitation='exp1 fold 0 only; validation used for diagnosis and F1 threshold selection'),indent=2))


if __name__ == '__main__': main()
