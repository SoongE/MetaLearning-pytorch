import os
from glob import glob
import numpy as np
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASEDIR = os.path.dirname(os.path.realpath(os.path.dirname(__file__)))


def extract_data(path, is_concat=True):
    runs = glob(os.path.join(BASEDIR, path, 'runs', '*/events.*'))

    if not len(runs):
        raise FileNotFoundError(f"{path} is not contain events.")
    df_list = []

    for exp in runs:
        x = EventAccumulator(path=exp)
        x.Reload()
        x.FirstEventTimestamp()
        tags = x.Tags()['scalars']  # ['Loss/Train', 'Loss/Val', 'Acc/Top1']

        steps = [e.step for e in x.Scalars(tags[0])]
        # wall_time = [e.wall_time for e in x.Scalars(tags[0])]
        # index = [e.index for e in x.Scalars(tags[0])]
        # count = [e.count for e in x.Scalars(tags[0])]
        n_steps = len(steps)

        data = np.zeros((n_steps, len(tags)))
        for i in range(len(tags)):
            data[:, i] = [e.value for e in x.Scalars(tags[i])]

        data_dict = {}
        for idx, tag in enumerate(tags):
            data_dict[tag] = data[:, idx]

        exp_name = os.path.basename(os.path.dirname(exp))
        data_dict['Name'] = [exp_name] * n_steps

        _df = pd.DataFrame(data=data_dict)

        if is_concat:
            df_list.append(_df)
        else:
            _df.to_csv(exp_name + '.csv', index_label='step')

    if is_concat:
        df = pd.concat(df_list)
        df.to_csv('Output.csv', index_label='step')


if __name__ == '__main__':
    extract_data('prototypical', False)
