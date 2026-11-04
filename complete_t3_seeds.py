"""Supplement sampling seeds 43/44 on the fixed seed-42 split and summarize."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_t3_sampling import run


def main():
    for experiment, folds in [('exp2',1),('exp1',5)]:
        root=Path('outputs/T3_sampling')/experiment
        original=pd.read_csv(root/'metrics.csv')
        original=original.loc[original.seed.eq(42)].copy()
        assert len(original)==folds*15
        original.to_csv(root/'seed_42_metrics.csv',index=False)
        frames=[original]
        for seed in (43,44):
            directory=root/f'seed_{seed}'
            print(f'START {experiment} seed={seed}',flush=True)
            frame=run(experiment,output_dir=directory,seed=seed,split_seed=42)
            for fold in range(folds):
                old=np.load(root/f'fold_{fold}'/'split_ids.npz')
                new=np.load(directory/f'fold_{fold}'/'split_ids.npz')
                for part in ('train','val','test'):
                    np.testing.assert_array_equal(old[f'{part}_row_ids'],new[f'{part}_row_ids'])
            frames.append(frame)
        combined=pd.concat(frames,ignore_index=True)
        assert len(combined)==folds*15*3
        assert not combined.duplicated(['fold','seed','mode']).any()
        assert combined.groupby(['fold','mode']).seed.nunique().eq(3).all()
        combined.to_csv(root/'metrics.csv',index=False)
        combined.to_csv(f'results/t3_sampling_{experiment}_metrics.csv',index=False)
        metrics=['average_precision','f1','roc_auc','precision','recall','fpr']
        within=combined.groupby(['fold','mode'])[metrics].agg(['mean','std'])
        within.columns=['_'.join(c) for c in within.columns]
        within.to_csv(f'results/t3_sampling_{experiment}_within_fold_seeds.csv')
        # Keep fold variation distinct from seed variation.
        if folds>1:
            summary=combined.groupby(['fold','mode'])[metrics].mean().groupby('mode').agg(['mean','std'])
        else:
            summary=combined.groupby('mode')[metrics].agg(['mean','std'])
        summary.columns=['_'.join(c) for c in summary.columns]
        summary.to_csv(root/'summary.csv')
        summary.to_csv(f'results/t3_sampling_{experiment}_summary.csv')
        protocol=dict(experiment=experiment,seeds=[42,43,44],split_seed=42,
            folds=folds,smote_ratio=.01,C=1,class_weight='balanced',evaluation='validation only',
            status='complete',artifacts='seed 42 in fold_*; seeds 43/44 in seed_<n>/fold_*',
            summary='exp1: seed-average within folds then cross-fold std; exp2: cross-seed std')
        (root/'run.json').write_text(json.dumps(protocol,indent=2))
        Path(f'results/t3_sampling_{experiment}_protocol.json').write_text(json.dumps(protocol,indent=2))
        print(f'COMPLETE {experiment}: {len(combined)} fits',flush=True)


if __name__=='__main__': main()
