import pandas as pd
import csv
import pickle
import argparse

from lentil import datatools
from lentil import models


def mnemosyneToTextfile(fname):
    with open(fname, 'rb') as f:
        history = pickle.load(f).data
    
    columns = ['module_id','outcome','timestamp','duration','timestep','module_type','user_id']
    file = open('mnemosyne.txt', 'w')
    file.write(','.join(columns)+'\n')
    for i in range(len(history)):
        line = ''
        for x in range(len(columns)):
            line = line + str(history.iloc[i][columns[x]])
            if x != (len(columns)-1):
                line += ","
        file.write(line + '\n')

def textToInteractionHistory(fname, timestamp, user_id, item_id, outcome, correct):
    data = []
    cols = ['outcome', 'timestamp','time_elapsed','student_id', 'module_id','module_type','timestep', 'history_seen', 'history_correct','exponential']
    f = open(fname, 'rb')

    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        temp = []
        p = False
        if row[outcome] == correct:
            p = True
        else:
            p = False
        temp.append(p)
        temp.append(int(row[timestamp]))
        temp.append(int(row['time_elapsed']))        
        temp.append(row[user_id])
        temp.append(row[item_id])
        temp.append('assessment')
        temp.append(1)
        temp.append(row['history_seen'])
        temp.append(row['history_correct'])
        temp.append(row['exponential'])

        data.append(temp)

    df = pd.DataFrame(columns = cols, data = data)
    ih = datatools.InteractionHistory(df)

    return ih
            

history = textToInteractionHistory('processed_data/mnemosyne_withfeatures.txt', 'timestamp', 'student_id','module_id','outcome', 'True')

# history = fileToHistory('data.txt')
# print history.data
# model = models.OneParameterLogisticModel(history.data, select_regularization_constant=True, name_of_user_id='student_id')
# model.fit()