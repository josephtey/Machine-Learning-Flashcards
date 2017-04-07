import numpy as np
import pandas as pd

def getData(df):
    data = {}
    for user_item_pair_id, group in df.groupby(['student_id', 'module_id']):
        if len(group) <= 1:
            continue
        timestamps = np.array(group['timestamp'].values)
        intervals = timestamps[1:] - timestamps[:-1]
        outcomes = np.array(group['outcome'].apply(lambda x: 1 if x else 0).values)
        data[user_item_pair_id] = (intervals, outcomes, timestamps)
    return data