import numpy as np
import pandas as pd
from run_new_exp1_tiers import run


def test_outer_modes_tiers_and_partitions(tmp_path):
    rng=np.random.default_rng(4)
    df=pd.DataFrame(rng.normal(size=(1200,28)),columns=[f'V{i}' for i in range(1,29)])
    df['Time']=np.arange(1200);df['Amount']=rng.lognormal(size=1200)
    df['Class']=(np.arange(1200)%20==0).astype(int)
    df.loc[df.Class.eq(1),'V14']-=5
    source=tmp_path/'data.csv';df.to_csv(source,index=False)
    output=tmp_path/'results';result=run(source,output)
    assert len(result)==72
    assert result.groupby(['mode','tier']).partition.nunique().eq(3).all()
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
