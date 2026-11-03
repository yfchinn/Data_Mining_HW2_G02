import numpy as np
import pandas as pd

from run_if_contamination import run


def test_pollution_is_included_without_changing_normal_population(tmp_path):
    rng=np.random.default_rng(7)
    frame=pd.DataFrame(rng.normal(size=(400,28)),columns=[f'V{i}' for i in range(1,29)])
    frame['Time']=np.arange(400)
    frame['Amount']=rng.lognormal(size=400)
    frame['Class']=(np.arange(400)%20==0).astype(int)
    frame.loc[frame.Class.eq(1),'V14']-=5
    source=tmp_path/'data.csv'
    frame.to_csv(source,index=False)
    result=run('exp2',source,tmp_path/'output')
    assert len(result)==15
    assert result.normal_count.nunique()==1
    assert result.loc[result['mode'].eq('C0'),'fraud_count'].eq(0).all()
    assert result.loc[result['mode'].eq('C0'),'delta_ap'].eq(0).all()
    assert result.loc[result['mode'].eq('C4'),'fraud_count'].gt(0).all()
    for _,group in result.groupby('seed'):
        assert group.fraud_count.is_monotonic_increasing
