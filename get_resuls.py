"""
Retrieve csv files produced by jobs from shared drive.
Make sure to edit RUNS_PATH. This is different on different operating systems.
"""
from typing import Optional
from pathlib import Path
import pandas as pd
import yaml
from collections import defaultdict

from ludwig.results import gen_param_paths

from semanticmodels2.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = None
RUNS_PATH = Path('/Volumes') / 'ludwig_data' / 'SemanticModels2' / 'runs'  # path to where runs are stored on shared drive

exp2label2accuracies = defaultdict(dict)
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         isolated=True if RUNS_PATH is not None else False,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         require_all_found=False,
                                         ):

    # get params
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)

    print(param2val)

    for p in param_path.rglob('pp_train.csv'):

        # read data
        df = pd.read_csv(p, index_col=0, squeeze=True)
        print(df)
