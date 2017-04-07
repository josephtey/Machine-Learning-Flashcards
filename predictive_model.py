"""
Module for probabilistic memory models
@author Siddharth Reddy <sgr45@cornell.edu>
"""

from __future__ import division

import sys

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import defaultdict, namedtuple

from matplotlib import pyplot as plt

from lentil import datatools
from lentil import models
import math

Instance = namedtuple('Instance', 'p t fv h ts uid eid expo'.split())

class EFCLinear(models.SkillModel):
    """
    Class for a memory model that predicts recall likelihood using basic statistics 
    of previous review intervals and outcomes for a user-item pair
    """

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
            if correct == None:
                if current_interaction['outcome'] == correct:
                    p = 0.75
                else:
                    p = 0.25
            else:
                p = pclip(float(current_interaction['outcome']))
            t = float(current_interaction['time_elapsed'])
            h = hclip(-t/(math.log(p, 2)))
            item_id = current_interaction['module_id']
            timestamp = int(current_interaction['timestamp'])
            user_id = current_interaction['student_id']
            seen = int(current_interaction['history_seen'])
            right = int(current_interaction['history_correct'])
            expo = float(current_interaction['exponential'])
            
            wrong = seen - right
            fv = []
            fv.append(math.sqrt(1+right))
            fv.append(math.sqrt(1+wrong))
            fv.append(math.sqrt(1+expo))
            inst = Instance(p, t, fv, h,timestamp, user_id, item_id, expo)
            instances.append(inst)
        
        #return list of instances
        return instances


    def fit(self, C=1.0):
        
        instances = self.extract_features(self.history, 'True')

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
        print X_train
        print Y_train
        self.clf = LinearRegression()
        self.clf.fit(X_train, Y_train)


    def assessment_pass_likelihoods(self, df):
        def predict(model, input, time_elapsed):
            h_power = model.predict(np.array(input).reshape(1,-1))
            h = math.pow(2, h_power)
            p = math.pow(2, (-time_elapsed)/h)
            return p
        predictions = np.array([])
        instances = self.extract_features(df, 'True')
        X_test = []
        for i in range(len(instances)):
            x = instances[i].fv
            x.append(instances[i].t)
            X_test.append(x)

        for i in range(len(X_test)):
            prediction = predict(self.clf, X_test[i][0:3],X_test[i][3])
            predictions = np.append(predictions, prediction)

        return predictions