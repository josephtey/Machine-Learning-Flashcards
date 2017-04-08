import pandas as pd
import numpy as np
import csv
import pickle
import argparse

from lentil import datatools
from lentil import models
import filterdata
from operator import itemgetter


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
        temp.append(int(row['history_seen']))
        temp.append(row['history_seen'])
        temp.append(row['history_correct'])
        temp.append(row['exponential'])

        data.append(temp)

    df = pd.DataFrame(columns = cols, data = data)
    ih = datatools.InteractionHistory(df)

    return ih

def filterFromArray(index, df):
    filtered = pd.DataFrame(columns=['outcome','timestamp','time_elapsed', 'student_id', 'module_id','module_type','timestep','history_seen','history_correct', 'exponential', 'time_since_previous_interaction','duration'])
    def comboIsInIndex(combo):
        x = False
        for i in range(len(index)):
            if index[i] == combo:
                x = True
                break;
        return x
    for i in range(len(df)):
        print str(i) + ' out of ' + str(len(df))
        combo = (df.iloc[i]['student_id'], df.iloc[i]['module_id'])
        if comboIsInIndex(combo) == True:
            filtered = filtered.append(df.iloc[i])
    return filtered

def filteredFromHistory(history):
    index = list(map(itemgetter(0), filterdata.getData(history).items()))
    return index

#unfiltered history
radical_hist = textToInteractionHistory('processed_data/radical_irt_2.txt', 'timestamp', 'user_id','item_id','p_recall', '0.75')
mnemo_hist = textToInteractionHistory('processed_data/mnemosyne_withfeatures.txt', 'timestamp', 'student_id','module_id','outcome', 'True')
spanish_hist = textToInteractionHistory('processed_data/spanish_processed.txt', 'timestamp', 'student_id','module_id','outcome', 'True')

#filtered histories
filtered_spanish = datatools.InteractionHistory(filterFromArray(filteredFromHistory(spanish_hist.data), spanish_hist.data))


