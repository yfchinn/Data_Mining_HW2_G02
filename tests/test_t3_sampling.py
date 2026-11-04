import json

import numpy as np
import pandas as pd
import pytest

from run_t3_sampling import run


@pytest.mark.parametrize('experiment', ['exp1', 'exp2'])
@pytest.mark.parametrize('test_ratio', [.2, .1])
def test_sampling_runner_preserves_validation(experiment, test_ratio, tmp_path):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(600, 28)), columns=[f'V{i}' for i in range(1,29)])
    frame['Time'] = np.arange(600)
    frame['Amount'] = rng.lognormal(size=600)
    frame['Class'] = (np.arange(600) % 20 == 0).astype(int)
    frame.loc[frame.Class.eq(1), 'V14'] -= 5
    source = tmp_path/'data.csv'
    frame.to_csv(source,index=False)
    output = tmp_path/'output'
    result = run(experiment, source, output, n_splits=2, smote_ratio=.1, test_ratio=test_ratio)
    assert len(result) == (30 if experiment == 'exp1' else 15)
    assert not list(output.rglob('*test_predictions*'))
    for fold, records in result.groupby('fold'):
        baseline = pd.read_csv(output/f'fold_{fold}'/'P0_val_predictions.csv')
        for mode in records['mode']:
            prediction = pd.read_csv(output/f'fold_{fold}'/f'{mode}_val_predictions.csv')
            pd.testing.assert_frame_equal(prediction[['row_id','y_true']],baseline[['row_id','y_true']])
        assert records.loc[records['mode'].str.startswith('P1'), 'training_rows'].max() < records.loc[records['mode'].eq('P0'), 'training_rows'].iloc[0]
    assert json.loads((output/'run.json').read_text())['status']=='complete'
    second=tmp_path/'seed43'
    run(experiment,source,second,n_splits=2,seed=43,smote_ratio=.1,split_seed=42,test_ratio=test_ratio)
    for fold in result.fold.unique():
        first_ids=np.load(output/f'fold_{fold}'/'split_ids.npz')
        assert len(first_ids['test_row_ids']) == round(600*test_ratio)
        assert sum(len(first_ids[f'{p}_row_ids']) for p in ('train','val','test'))==600
        second_ids=np.load(second/f'fold_{fold}'/'split_ids.npz')
        for partition in ('train','val','test'):
            np.testing.assert_array_equal(first_ids[f'{partition}_row_ids'],second_ids[f'{partition}_row_ids'])
