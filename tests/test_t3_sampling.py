import json

import numpy as np
import pandas as pd
import pytest

from run_t3_sampling import run


@pytest.mark.parametrize('experiment', ['exp1', 'exp2'])
def test_sampling_runner_preserves_validation(experiment, tmp_path):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(600, 28)), columns=[f'V{i}' for i in range(1,29)])
    frame['Time'] = np.arange(600)
    frame['Amount'] = rng.lognormal(size=600)
    frame['Class'] = (np.arange(600) % 20 == 0).astype(int)
    frame.loc[frame.Class.eq(1), 'V14'] -= 5
    source = tmp_path/'data.csv'
    frame.to_csv(source,index=False)
    output = tmp_path/'output'
    result = run(experiment, source, output, n_splits=2, smote_ratio=.1)
    assert len(result) == (30 if experiment == 'exp1' else 15)
    assert not list(output.rglob('*test_predictions*'))
    for fold, records in result.groupby('fold'):
        baseline = pd.read_csv(output/f'fold_{fold}'/'P0_val_predictions.csv')
        for mode in records['mode']:
            prediction = pd.read_csv(output/f'fold_{fold}'/f'{mode}_val_predictions.csv')
            pd.testing.assert_frame_equal(prediction[['row_id','y_true']],baseline[['row_id','y_true']])
        assert records.loc[records['mode'].str.startswith('P1'), 'training_rows'].max() < records.loc[records['mode'].eq('P0'), 'training_rows'].iloc[0]
    assert json.loads((output/'run.json').read_text())['status']=='complete'
