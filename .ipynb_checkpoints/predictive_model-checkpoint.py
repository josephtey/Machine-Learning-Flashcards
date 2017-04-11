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
import evaluate as e
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
            t = (float(current_interaction['time_elapsed']))
            h = hclip(-t/(math.log(p, 2)))
            item_id = current_interaction['module_id']
            timestamp = int(current_interaction['timestamp'])
            user_id = current_interaction['student_id']
            
            fv = []
            for x in range(len(constants.FEATURE_NAMES)):
                if isfloat(current_interaction[constants.FEATURE_NAMES[x]]):
                    if constants.FEATURE_NAMES[x] == 'activation':
                        fv.append(float(current_interaction[constants.FEATURE_NAMES[x]]))
                    else:
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
        
    def predict(self, model, input, time_elapsed):
            h_power = model.predict(np.array(input).reshape(1,-1))
            h = math.pow(2, h_power)
            p = math.pow(2, (-time_elapsed)/h)
            return p
        
    def assessment_pass_likelihoods(self, df):
        
        predictions = np.array([])
        instances = self.extract_features(df, correct=True)
        X_test = []
        for i in range(len(instances)):
            x = instances[i].fv
            x.append(instances[i].t)
            X_test.append(x)
        for i in range(len(X_test)):
            prediction = self.predict(self.clf, X_test[i][0:len(constants.FEATURE_NAMES)],X_test[i][len(constants.FEATURE_NAMES)])
            predictions = np.append(predictions, prediction)

        return predictions

    
class LogisticRegressionModel(models.SkillModel):

    def __init__(self, history, name_of_user_id='user_id'):

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id
        self.clf = None
    
    def get_features(self, df):
        fv = []
        for i in range(len(constants.FEATURE_NAMES)):
            feature = df[constants.FEATURE_NAMES[i]]
            if isfloat(feature):
                fv.append(float(feature))
            elif isint(feature):
                fv.append(int(feature))
        return fv
        
    def fit(self, C=1.0):
        X_train = []
        Y_train = []
        for x in range(len(self.history)):
            X_train.append(self.get_features(self.history.iloc[x]))
            Y_train.append(self.history.iloc[x]['outcome'])
        
        self.clf = LogisticRegression(C=C)
        self.clf.fit(X_train, Y_train)
        
    def predict(self, inputs):
        output = self.clf.predict_proba(np.array(inputs).reshape(1,-1))[:,1]
        return output
    
    def assessment_pass_likelihoods(self, df):
        inputs = []
        for i in range(len(df)):
            inputs.append(self.get_features(self.history.iloc[i]))
            
        return self.clf.predict_proba(inputs)[:, 1]   

    
    
    
    
    