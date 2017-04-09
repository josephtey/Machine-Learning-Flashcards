from __future__ import division

import sys

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import numpy as np
from collections import defaultdict, namedtuple

from matplotlib import pyplot as plt

from lentil import datatools
from lentil import models
from lentil import evaluate 

import math
import tools as t
import constants

Instance = namedtuple('Instance', 'p t fv h ts uid eid'.split())

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

class EFCLinear(models.SkillModel):

    def __init__(self, history, name_of_user_id='user_id'):


        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id

        self.clf = None
        self.data = None

    def extract_features(self, df, correct=None):

        def pclip(p):
            # bound min/max model predictions (helps with loss optimization)
            return min(max(p, 0.0001), .9999)

        def hclip(h):
            # bound min/max half-life
            return min(max(h, 5), 2628000)

        instances = []
        for i in range(len(df)):
            current_interaction = df.iloc[i]
            p = pclip(float(current_interaction['outcome']))
            t = float(current_interaction['time_elapsed'])
            h = hclip(-t/(math.log(p, 2)))
            item_id = current_interaction['module_id']
            timestamp = int(current_interaction['timestamp'])
            user_id = current_interaction['student_id']
            
            fv = []
            for x in range(len(constants.FEATURE_NAMES)):
                if isfloat(current_interaction[constants.FEATURE_NAMES[x]]):
                    fv.append(math.sqrt(1.0+float(current_interaction[constants.FEATURE_NAMES[x]])))
                elif isint(current_interaction[constants.FEATURE_NAMES[x]]):
                    fv.append(math.sqrt(1+int(current_interaction[constants.FEATURE_NAMES[x]]))) 
            
            inst = Instance(p, t, fv, h,timestamp, user_id, item_id)
            instances.append(inst)
        
        #return list of instances
        return instances


    def fit(self, C=1.0):
        
        instances = self.extract_features(self.history, correct=True)

        X_train = []
        Y_train = []
        for i in range(len(instances)):
            x = []
            #inputs
            x.append(instances[i].fv)
            #outputs
            Y_train.append(math.log(instances[i].h, 2))

            #append inputs and outputs to training lists
            X_train.append(x[0])
            
        self.clf = LinearRegression()
        self.clf.fit(X_train, Y_train)


    def assessment_pass_likelihoods(self, df):
        def predict(model, input, time_elapsed):
            h_power = model.predict(np.array(input).reshape(1,-1))
            h = math.pow(2, h_power)
            p = math.pow(2, (-time_elapsed)/h)
            return p
        
        predictions = np.array([])
        instances = self.extract_features(df, correct=True)
        X_test = []
        for i in range(len(instances)):
            x = instances[i].fv
            x.append(instances[i].t)
            X_test.append(x)
        for i in range(len(X_test)):
            prediction = predict(self.clf, X_test[i][0:len(constants.FEATURE_NAMES)],X_test[i][len(constants.FEATURE_NAMES)])
            predictions = np.append(predictions, prediction)

        return predictions

class LogisticRegressionModel(models.SkillModel):

    def __init__(self, history, name_of_user_id='user_id'):

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id

        self.clf = None
        self.data = None

    def extract_features(self, review_history, max_time=None):

        intervals, outcomes, timestamps = review_history

        if max_time is not None:
            # truncate the sequences 
            i = 1
            while i < len(timestamps) and timestamps[i] <= max_time:
                i += 1
            outcomes = outcomes[:i-1]
            intervals = intervals[:i-1]

        if len(intervals) == 0:
            interval_feature_list = [0] * 8
        else:
            intervals = np.log(np.array(intervals)+1)
            interval_feature_list = [len(intervals), intervals[0], intervals[-1], \
                    np.mean(intervals), min(intervals), max(intervals), \
                    max(intervals)-min(intervals), sorted(intervals)[len(intervals) // 2]]

        if len(outcomes) == 0:
            outcome_feature_list = [0] * 8
        else:
            outcome_feature_list = [len(outcomes), outcomes[0], outcomes[-1], \
                    np.mean(outcomes), min(outcomes), max(outcomes), \
                    max(outcomes)-min(outcomes), sorted(outcomes)[len(outcomes) // 2]]

        return np.array(interval_feature_list + outcome_feature_list)

    def fit(self, C=1.0):
        
        self.data = {}
        for user_item_pair_id, group in self.history.groupby([self.name_of_user_id, 'module_id']):
            if len(group) <= 1:
                continue
            timestamps = np.array(group['timestamp'].values)
            intervals = timestamps[1:] - timestamps[:-1]
            outcomes = np.array(group['outcome'].apply(lambda x: 1 if x else 0).values)
            self.data[user_item_pair_id] = (intervals, outcomes, timestamps)

        X_train = np.array([self.extract_features(
            (intervals[:i+1], outcomes[:i+1], timestamps[:i+1])) \
                for intervals, outcomes, timestamps in self.data.itervalues() \
                for i in xrange(len(intervals))])
        Y_train = np.array([x for intervals, outcomes, timestamps in self.data.itervalues() \
                for x in outcomes[1:]])

        self.clf = LogisticRegression(C=C)
        self.clf.fit(X_train, Y_train)

    def assessment_pass_likelihoods(self, df):
        X = np.array([self.extract_features(self.data[(x[self.name_of_user_id], 
            x['module_id'])], max_time=x['timestamp']) for _, x in df.iterrows()])
        return self.clf.predict_proba(X)[:,1]    

