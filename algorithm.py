import argparse
import csv
import gzip
import math
import os
import random
import sys
import numpy as np
import random
import pandas as pd

from sklearn import linear_model
from sklearn.externals import joblib
from collections import defaultdict, namedtuple
from lentil import models
from lentil import evaluate 
from lentil import datatools


Instance = namedtuple('Instance', 'p t fv h a ts uid eid expo'.split())

def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)

def hclip(h):
    # bound min/max half-life
    return min(max(h, 5), 2628000)

def read_data(input_file, max_lines=None):
    # read learning trace data in specified format, see README for details
    print 'reading data...'
    instances = list()
    f = open(input_file, 'rb')

    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if args.l == False:
            if row[args.outcome_id] == args.correct:
                p = 0.75
            else:
                p = 0.25
        else:
            p = pclip(float(row[args.outcome_id]))
        t = float(row[args.timeelapsed_id])
        # convert time delta to days
        h = hclip(-t/(math.log(p, 2)))
        item_id = row[args.module_id]
        timestamp = int(row[args.timestamp_id])
        user_id = row[args.user_id]
        seen = int(row['history_seen'])
        right = int(row['history_correct'])
        expo = float(row['exponential'])
       
        wrong = seen - right
        # feature vector is a list of (feature, value) tuples
        fv = []
        fv.append((intern('right'), math.sqrt(1+right)))
        fv.append((intern('wrong'), math.sqrt(1+wrong)))
        fv.append((intern('expo'), math.sqrt(1+expo)))
        inst = Instance(p, t, fv, h, (right+2.)/(seen+4.),timestamp, user_id, item_id, expo)
        instances.append(inst)

    print 'done!'
    splitpoint = int(0.9 * len(instances))
    return instances[:splitpoint], instances[splitpoint:]

def instToArray(instances, ret='h'):
    output_x = []
    output_y = []
    for i in range(len(instances)):
        x = []
        x.append(instances[i].fv[0][1])
        x.append(instances[i].fv[1][1])
        x.append(instances[i].fv[2][1])

        if ret == 'p':
            x.append(instances[i].t)

        output_x.append(x)
        if ret == 'h':
            output_y.append(math.log(instances[i].h, 2))
        else:
            output_y.append(round(instances[i].p))
    return output_x, output_y

def instToDataFrame(instances):
    data = []
    cols = ['outcome', 'timestamp', 'student_id', 'module_id','module_type','timestep']
    for i in range(len(instances)):
        temp = []
        p = False
        if instances[i].p == 0.75:
            p = True
        else:
            p = False
        temp.append(p)
        temp.append(instances[i].ts)
        temp.append(instances[i].uid)
        temp.append(instances[i].eid)
        temp.append('assessment')
        temp.append(1)

        data.append(temp)
    df = pd.DataFrame(columns = cols, data = data)
    ih = datatools.InteractionHistory(df)

    return ih

def predict(model, input, time_elapsed):
    h_power = model.predict(np.array(input).reshape(1,-1))
    h = math.pow(2, h_power)
    p = math.pow(2, (-time_elapsed)/h)
    return p

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def evaluate_model(model, x, y):
    correct = 0
    wrong = 0
    if model == 'random':
        for i in range(len(y)):
            prediction = random.randrange(0,2)
            if prediction == y[i]:
                correct += 1
            else:
                wrong += 1
    else: 
        for i in range(len(x)):
            fv = [x[i][0], x[i][1], x[i][2]]
            t = x[i][3]
            prediction = round(predict(model, fv, t))
            if prediction == y[i]:
                correct += 1

            else:
                wrong += 1

    total = float(correct) + float(wrong)
    return float(correct)/total

argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('input_file', action="store", help='log file for training')
argparser.add_argument('-user_id', action="store", dest="user_id", type=str, default='user_id')
argparser.add_argument('-module_id', action="store", dest="module_id", type=str, default='item_id')
argparser.add_argument('-outcome_id', action="store", dest="outcome_id", type=str, default='outcome')
argparser.add_argument('-timestamp_id', action="store", dest="timestamp_id", type=str, default='timestamp')
argparser.add_argument('-timeelapsed_id', action="store", dest="timeelapsed_id", type=str, default='time_elapsed')
argparser.add_argument('-correct_id', action="store", dest="correct", type=str, default='CORRECT')
argparser.add_argument('-l', action="store_true", default=False)

args = argparser.parse_args()

train_data = read_data(args.input_file)[0]
train_data_X = instToArray(train_data)[0]
train_data_Y = instToArray(train_data)[1]

test_data = read_data(args.input_file)[1]
test_data_p_x = instToArray(test_data, 'p')[0]
test_data_p_y = instToArray(test_data, 'p')[1]
print train_data_X
print train_data_Y

OnePlIRTModel = models.OneParameterLogisticModel(instToDataFrame(train_data).data, select_regularization_constant=True, name_of_user_id='student_id')
OnePlIRTModel.fit()
# correct = 0
# wrong = 0
# for i in range(len(instToDataFrame(train_data).data)):
#     prediction = OnePlIRTModel.assessment_pass_likelihood(instToDataFrame(train_data).data.iloc[i])
#     print prediction
#     if round(prediction) == instToDataFrame(train_data).data.iloc[i]['outcome']:
#         correct += 1
#     else:
#         wrong += 1
#     print correct, wrong
# total = float(correct) + float(wrong)
# print float(correct)/total

SpacedRepetitionModel = linear_model.LinearRegression()
SpacedRepetitionModel.fit(train_data_X, train_data_Y)
print evaluate_model(SpacedRepetitionModel, test_data_p_x, test_data_p_y)

joblib.dump(SpacedRepetitionModel, 'models/model.pkl', compress=9)

