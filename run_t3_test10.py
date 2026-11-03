"""Exploratory split change after original test evaluation; validation only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_t3_sampling import run


def main():
    for exp,folds in [('exp2',1),('exp1',5)]:
        root=Path('outputs/T3_test10')/exp
        frames=[]
        for seed in (42,43,44):
            print('START test10',exp,seed,flush=True)
            directory=root/f'seed_{seed}'
            frame=run(exp,output_dir=directory,seed=seed,split_seed=42,test_ratio=.1)
            for fold in range(folds):
                reference=np.load(root/'seed_42'/f'fold_{fold}'/'split_ids.npz')
                current=np.load(directory/f'fold_{fold}'/'split_ids.npz')
                for part in ('train','val','test'):
                    np.testing.assert_array_equal(reference[f'{part}_row_ids'],current[f'{part}_row_ids'])
            frames.append(frame)
        frame=pd.concat(frames,ignore_index=True)
        assert len(frame)==folds*45
        assert frame.groupby(['fold','mode']).seed.nunique().eq(3).all()
        frame.to_csv(f'results/t3_test10_{exp}_metrics.csv',index=False)
        metrics=['average_precision','f1','roc_auc','precision','recall','fpr']
        within=frame.groupby(['fold','mode'])[metrics].agg(['mean','std'])
        within.columns=['_'.join(c) for c in within.columns]
        within.to_csv(f'results/t3_test10_{exp}_within_fold_seeds.csv')
        if folds>1: summary=frame.groupby(['fold','mode'])[metrics].mean().groupby('mode').agg(['mean','std'])
        else: summary=frame.groupby('mode')[metrics].agg(['mean','std'])
        summary.columns=['_'.join(c) for c in summary.columns]
        summary.to_csv(f'results/t3_test10_{exp}_summary.csv')
        protocol=dict(experiment=exp,seeds=[42,43,44],split_seed=42,test_ratio=.1,
            split='72/18/10 stratified folds' if exp=='exp1' else '70/20/10 chronological',
            class_weight='balanced',C=1,smote_ratio=.01,evaluation='validation only',
            stage='post-test exploratory; validation populations change; not an independent confirmation',
            status='complete')
        root.mkdir(parents=True,exist_ok=True)
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
        Path(f'results/t3_test10_{exp}_protocol.json').write_text(json.dumps(protocol,indent=2))
        print('COMPLETE',exp,len(frame),'fits',flush=True)


if __name__=='__main__': main()
