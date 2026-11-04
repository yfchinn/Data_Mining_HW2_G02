import numpy as np
import pandas as pd
import joblib

from run_new_exp1_if_contamination import run


def test_pollution_is_fitted_and_evaluation_is_fixed(tmp_path):
    rng=np.random.default_rng(12)
    frame=pd.DataFrame(rng.normal(size=(1200,28)),columns=[f'V{i}' for i in range(1,29)])
    frame['Time']=np.arange(1200)
    frame['Amount']=rng.lognormal(size=1200)
    frame['Class']=(np.arange(1200)%20==0).astype(int)
    frame.loc[frame.Class.eq(1),'V14']-=5
    source=tmp_path/'data.csv'
    frame.to_csv(source,index=False)
    output=tmp_path/'output'
    result=run(source,output,seeds=(42,))
    assert len(result)==15
    previous=set()
    baseline=None
    for mode in ('C0','C1','C2','C3','C4'):
        directory=output/'seed_42'/mode
        ids=np.load(directory/'fit_row_ids.npy')
        fraud=set(ids[frame.iloc[ids]['Class'].to_numpy()==1])
        assert previous <= fraud
        previous=fraud
        model=joblib.load(directory/'T2.joblib')['model']
        assert model._n_samples==len(ids)
        prediction=pd.read_csv(directory/'test_predictions.csv')
        if baseline is None: baseline=prediction[['row_id','y_true']]
        pd.testing.assert_frame_equal(prediction[['row_id','y_true']],baseline)
        rows=result[result['mode'].eq(mode)]
        assert rows.threshold.nunique()==1
    assert len(previous)>0
    assert result[result['mode'].eq('C0')].delta_ap.eq(0).all()
