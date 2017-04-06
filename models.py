import pandas as pd
import csv
import pickle

from lentil import datatools
from lentil import models


def mnemosyneToTextfile(fname):
    with open((fname), 'rb') as f:
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

mnemosyneToTextfile('../lentil/mnemosyne_history.pkl')
            
def fileToHistory(fname):
    file = open(fname, 'r')
    items = csv.reader(file)
    raw_list = list(items)
    cols = raw_list[0]
    data = raw_list[1:]
    #convert strings to float and int values
    for i in range(len(data)):
        data[i][0] = bool(data[i][0])
        data[i][1] = int(data[i][1])                            
        data[i][2] = int(data[i][2])
        data[i][3] = int(data[i][3])
        data[i][4] = int(data[i][4])
        data[i][5] = int(data[i][5])
        data[i][6] = str(data[i][6])
        data[i][7] = int(data[i][7])
        data[i][8] = int(data[i][8])
        data[i][9] = float(data[i][9])

    df = pd.DataFrame(columns = cols, data = data)
    ih = datatools.InteractionHistory(df)

    return ih


# history = fileToHistory('data.txt')
# print history.data
# model = models.OneParameterLogisticModel(history.data, select_regularization_constant=True, name_of_user_id='student_id')
# model.fit()