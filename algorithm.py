import argparse
import csv
import gzip
import math
import os
import random
import sys
import numpy as np
import random

from sklearn import linear_model
from sklearn.externals import joblib
from collections import defaultdict, namedtuple


Instance = namedtuple('Instance', 'p t fv h a ts uid eid expo'.split())

def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)


def read_data(input_file, max_lines=None):
    # read learning trace data in specified format, see README for details
    print 'reading data...'
    instances = list()
    f = open(input_file, 'rb')

    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if max_lines is not None and i >= max_lines:
            break
        p = pclip(float(row['p_recall']))
        t = float(row['time_elapsed'])
        # convert time delta to days
        h = -t/(math.log(p, 2))
        item_id = row['item_id']
        timestamp = int(row['timestamp'])
        user_id = row['user_id']
        seen = int(row['history_seen'])
        right = int(row['history_correct'])
        expo = float(row['exponential'])
        item_difficulty = float(row['item_difficulty'])
        user_ability = float(row['user_ability'])
       
        wrong = seen - right
        # feature vector is a list of (feature, value) tuples
        fv = []
        fv.append((intern('right'), right))
        fv.append((intern('wrong'), wrong))
        fv.append((intern('expo'), expo))
        fv.append((intern('item_difficulty'), item_difficulty))
        fv.append((intern('user_ability'), user_ability))
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
        x.append(instances[i].fv[3][1])
        x.append(instances[i].fv[4][1])

        if ret == 'p':
            x.append(instances[i].t)

        output_x.append(x)

        if ret == 'h':
            output_y.append(math.log(instances[i].h, 2))
        else:
            output_y.append(round(instances[i].p))
    return output_x, output_y

def predict(model, input, time_elapsed):
    h_power = model.predict(input)
    h = math.pow(2, h_power)
    p = math.pow(2, (-time_elapsed)/h)
    return p

def evaluate(model, x, y):
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
            fv = [x[i][0], x[i][1], x[i][2], x[i][3], x[i][4]]
            t = x[i][5]
            prediction = round(predict(model, fv, t))
            if prediction == y[i]:
                correct += 1
            else:
                wrong += 1

    total = float(correct) + float(wrong)
    return float(correct)/total

argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('input_file', action="store", help='log file for training')
args = argparser.parse_args()

train_data = read_data(args.input_file)[0]
train_data_X = instToArray(train_data)[0]
train_data_Y = instToArray(train_data)[1]

test_data = read_data(args.input_file)[1]
# test_data_X = instToArray(test_data)[0]
# test_data_Y = instToArray(test_data)[1]
test_data_p_x = instToArray(test_data, 'p')[0]
test_data_p_y = instToArray(test_data, 'p')[1]

SpacedRepetitionModel = linear_model.LinearRegression()
SpacedRepetitionModel.fit(train_data_X, train_data_Y)

print evaluate(SpacedRepetitionModel, test_data_p_x, test_data_p_y)

joblib.dump(SpacedRepetitionModel, 'models/model.pkl', compress=9)

