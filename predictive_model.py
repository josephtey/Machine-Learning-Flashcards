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

Instance = namedtuple('Instance', 'p t fv h ts uid eid expo'.split())

class EFCLinear(models.SkillModel):
    """
    :)
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
            prediction = predict(self.clf, X_test[i][0:3],X_test[i][3])
            predictions = np.append(predictions, prediction)

        return predictions

class LogisticRegressionModel(models.SkillModel):
    """
    Class for a memory model that predicts recall likelihood using basic statistics 
    of previous review intervals and outcomes for a user-item pair
    """

    def __init__(self, history, name_of_user_id='user_id'):
        """
        Initialize memory model object
        :param pd.DataFrame history: Interaction log data. Must contain the 'tlast' column,
            in addition to the other columns that belong to the dataframe in a
            lentil.datatools.InteractionHistory object. If strength_model is not None, then
            the history should also contain a column named by the strength_model (e.g., 'nreps' or
            'deck'). Rows should be sorted in increasing order of timestamp.
        :param str name_of_user_id: Name of column in history that stores user IDs (useful for
            distinguishing between user IDs and user-item pair IDs)
        """

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id

        self.clf = None
        self.data = None

    def extract_features(self, review_history, max_time=None):
        """
        Map a sequence of review intervals and outcomes to a fixed-length feature set
        :param (np.array,np.array,np.array) review_history: A tuple of 
            (intervals, outcomes, timestamps) where intervals are the milliseconds elapsed between 
            consecutive reviews, outcomes are binary and timestamps are unix epochs. Note that 
            there is one fewer element in the intervals array than in the outcomes and timestamps.
        :param int max_time: Intervals that occur after this time should not be used to 
            construct the feature set. Outcomes that occur at or after this time should not be
            used in the feature set either.
        :rtype: np.array
        :return: A feature vector for the review history containing the length, first, last,
            mean, min, max, range, and median (in that order) of the log-intervals, concatenated 
            with the length, first, last, mean, min, max, range, and median (in that order) of the 
            outcomes.
        """

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
        """
        Estimate the coefficients of a logistic regression model with a bias term and an L2 penalty
        
        :param float C: Regularization constant. Inverse of regularization strength.
        """
        
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
        """
        Compute recall likelihoods given the learned coefficients
        :param pd.DataFrame df: Interaction log data
        :rtype: np.array
        :return: An array of recall likelihoods
        """
       
        X = np.array([self.extract_features(self.data[(x[self.name_of_user_id], 
            x['module_id'])], max_time=x['timestamp']) for _, x in df.iterrows()])
        return self.clf.predict_proba(X)[:,1]    
