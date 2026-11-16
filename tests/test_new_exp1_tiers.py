import numpy as np
import pandas as pd
from run_new_exp1_tiers import run


def test_outer_modes_tiers_and_partitions(tmp_path):
    rng=np.random.default_rng(4)
    df=pd.DataFrame(rng.normal(size=(10000,28)),columns=[f'V{i}' for i in range(1,29)])
    df['Time']=np.arange(10000);df['Amount']=rng.lognormal(size=10000)
    df['Class']=(np.arange(10000)<492).astype(int)
    df.loc[df.Class.eq(1),'V14']-=5
    source=tmp_path/'data.csv';df.to_csv(source,index=False)
    output=tmp_path/'results';result=run(source,output)
    assert len(result)==72
    assert result.groupby(['mode','tier']).partition.nunique().eq(3).all()
    for modes in [('P1-1','P2-1'), ('P1-5','P2-5'),
                  ('P1-10','P2-10','P3-10','P4-10')]:
        reference=np.load(output/modes[0]/'split_ids.npz')
        for mode in modes[1:]:
            actual=np.load(output/mode/'split_ids.npz')
            for part in ('val','test'):
                np.testing.assert_array_equal(actual[f'{part}_row_ids'],reference[f'{part}_row_ids'])
    counts=pd.read_csv(output/'split_counts.csv').set_index(['mode','partition'])
    assert tuple(counts.loc[('P1-10','test'), ['normal','fraud']]) == (984,99)
    assert tuple(counts.loc[('P3-10','train'), ['normal','fraud']]) == (1475,295)
    for mode in result['mode'].unique():
        directory=output/mode
        ids=np.load(directory/'split_ids.npz')
        assert set(ids['train_row_ids']).isdisjoint(ids['test_row_ids'])
        for part in ('val','test'):
            expected=None
            for tier in ('T1','T2','T3'):
                p=pd.read_csv(directory/f'{tier}_{part}_predictions.csv')
                if expected is not None: pd.testing.assert_frame_equal(p[['row_id','y_true']],expected)
                expected=p[['row_id','y_true']]
        for tier in ('T2','T3'):
            thresholds=result.loc[(result['mode']==mode)&result.tier.eq(tier),'threshold']
            assert thresholds.nunique()==1
