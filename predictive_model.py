from __future__ import division

import sys

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from autograd import grad

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
import random

MIN_HALF_LIFE = 5   
MAX_HALF_LIFE = 2628000             
LN2 = math.log(2.)

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
    
def sclip(s):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(s, 0.20), .80)

def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 0.0001), .9999)

def hclip(h):
    # bound min/max half-life
    return min(max(h, 5), 2628000)

class EFC(models.SkillModel):
    def __init__(self, history, name_of_user_id='user_id', initial_weights=None, lrate=.001, hlwt=.01, l2wt=.1, sigma=1.):
        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id
        
        self.weights = defaultdict(float)
        if initial_weights is not None:
            self.weights.update(initial_weights)
        self.fcounts = defaultdict(int)
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma
        self.clf = None
        
    def extract_features(self, df, correct=None):

        instances = []
        for i in range(len(df)):
            current_interaction = df.iloc[i]
            
            p = sclip(float(current_interaction['outcome']))
            t = float(current_interaction['time_elapsed'])
            h = hclip(-t/(math.log(p, 2)))
            item_id = current_interaction['module_id']
            timestamp = int(current_interaction['timestamp'])
            user_id = current_interaction['student_id']
            
            fv = []
            for x in range(2):
                feature = current_interaction[constants.FEATURE_NAMES[x]]
                if isfloat(feature):
                    fv.append((constants.FEATURE_NAMES[x], math.sqrt(1.0+float(feature)) ))
                elif isint(feature):
                    fv.append((constants.FEATURE_NAMES[x], math.sqrt(1+int(feature))))

            inst = Instance(p, t, fv, h, timestamp, user_id, item_id)
            instances.append(inst)
        
        #return list of instances
        return instances
    def train_update(self, inst):
        base = 2.
        p, h = self.predict(inst, base)
        dlp_dw = 2.*(p-inst.p)*(LN2**2)*p*(inst.t/h)
        dlh_dw = 2.*(h-inst.h)*LN2*h
        for (k, x_k) in inst.fv:
            rate = (1./(1+inst.p)) * self.lrate / math.sqrt(1 + self.fcounts[k])
            self.weights[k] -= rate * dlp_dw * x_k
            self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
            self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2
            self.fcounts[k] += 1
        
    def fit(self, C=1.0):
        
        instances = self.extract_features(self.history, correct=True)

        for inst in instances:
            #print inst
            self.train_update(inst)
        
        print self.weights
            
    def halflife(self, inst, base):
        try:
            dp = sum([self.weights[k]*x_k for (k, x_k) in inst.fv])
            return hclip(base ** dp)
        except:
            return MAX_HALF_LIFE   
        
    def predict(self, inst, base=2.):
        h = self.halflife(inst, base)
        p = 2. ** (-inst.t/h)
        print h
        return pclip(p), h
                  
        
    def assessment_pass_likelihoods(self, df):
        instances = self.extract_features(df, correct=True)
        predictions = []
        for inst in instances:
            predictions.append(self.predict(inst))

        return predictions
    
class EFCLinear(models.SkillModel):

    def __init__(self, history, name_of_user_id='user_id', using_delay=True, strength_var='ml', abilities=None, difficulties=None):
        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id

        self.clf = None
        self.using_delay = using_delay
        self.strength_var = strength_var
        self.abilities = abilities
        self.difficulties = difficulties
        if self.strength_var == 'ml':
            self.FEATURE_NAMES =  ['seen', 'history_correct', 'history_wrong', 'exponential', 'wrong_streak', 'right_streak', 'average_outcome', 'average_time']
        elif self.strength_var == 'numreviews':
            self.FEATURE_NAMES = ['seen']
        elif self.strength_var == 'settles' or self.strength_var == 'leitner':
            self.FEATURE_NAMES = ['history_correct', 'history_wrong']
        elif self.strength_var == 'expo':
            self.FEATURE_NAMES =  ['exponential']

    def extract_features(self, df, correct=None):
        
        if self.abilities != None:
            users = list(set([str(i) for i in list(df['student_id'])]))
            abilities = dict(zip(users, self.abilities))
        
        if self.difficulties != None:
            items = list(set([str(i) for i in list(df['module_id'])]))
            difficulties = dict(zip(items, self.difficulties))
        
        instances = []
        for i in range(len(df)):
            current_interaction = df.iloc[i]
            
            p = sclip(float(current_interaction['outcome']))
            t = float(current_interaction['time_elapsed'])
            h = hclip(-t/(math.log(p, 2)))
            item_id = current_interaction['module_id']
            timestamp = int(current_interaction['timestamp'])
            user_id = current_interaction['student_id']
            
            
            fv = []
            for x in range(len(self.FEATURE_NAMES)):
                if self.FEATURE_NAMES[x] != 'seen':
                    feature = current_interaction[self.FEATURE_NAMES[x]]
                    if isfloat(feature):
                        fv.append(math.sqrt(1.0+float(feature)))
                    elif isint(feature):
                        fv.append(math.sqrt(1+int(feature)))
                else:
                    fv.append(math.sqrt(((int(current_interaction['history_correct'])**2)-1) + ((int(current_interaction['history_wrong'])**2)-1) + 1))
            
            #IRT parameters
            if self.abilities != None:
                ability = abilities[user_id]
                fv.append(ability)
            
            if self.difficulties != None:
                difficulty = difficulties[item_id]
                fv.append(difficulty)
            
            
            inst = Instance(p, t, fv, h, timestamp, user_id, item_id)
            instances.append(inst)
        
        #return list of instances
        return instances


    def fit(self, C=1.0):
        
        instances = self.extract_features(self.history, correct=True)

        X_train = []
        Y_train = []
        for i in range(len(instances)):
            #outputs
            Y_train.append(math.log(instances[i].h, 2))

            #append inputs and outputs to training lists
            X_train.append(instances[i].fv)

        
        self.clf = LinearRegression()
        self.clf.fit(X_train, Y_train)
        
    def predict(self, model, input, time_elapsed):
        h_power = model.predict(np.array(input).reshape(1,-1))            
        h = hclip(math.pow(2,h_power))

        #if self.strength_var == 'expo':
        #    h = hclip(math.exp(expo))
        #elif self.strength_var == 'numreviews':
        #    h = hclip(math.exp(seen))
        #elif self.strength_var == 'correct':
        #    h = correct*500
        if self.strength_var == 'leitner':
            h = hclip(math.exp(input[0]))
                
        try:
            p = pclip(math.pow(2, (-time_elapsed)/h))
        except:
            p = 0
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
            if self.abilities != None and self.difficulties == None or self.difficulties != None and self.abilities == None:
                prediction = self.predict(self.clf, X_test[i][0:len(self.FEATURE_NAMES)+2],X_test[i][2+len(self.FEATURE_NAMES)])
            elif self.abilities != None and self.difficulties != None:
                prediction = self.predict(self.clf, X_test[i][0:len(self.FEATURE_NAMES)+2],X_test[i][2+len(self.FEATURE_NAMES)])
            else:
                prediction = self.predict(self.clf, X_test[i][0:len(self.FEATURE_NAMES)],X_test[i][len(self.FEATURE_NAMES)])
            predictions = np.append(predictions, prediction)

        return predictions

    
class LogisticRegressionModel(models.SkillModel):

    def __init__(self, history, name_of_user_id='user_id', using_time=False):

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id
        self.clf = None
        self.using_time = using_time
    
    def get_features(self, df):
        fv = []
        for i in range(3):
            feature = df[constants.FEATURE_NAMES[i]]
            if isfloat(feature):
                fv.append(math.sqrt(1.0+float(feature)))
            elif isint(feature):
                fv.append(math.sqrt(1+int(feature)))
        if self.using_time:
            fv.append(df['time_elapsed'])
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
            history = df.iloc[i]
            inputs.append(self.get_features(history))
           
        return self.clf.predict_proba(inputs)[:, 1]   
    
class PercentageModel(models.SkillModel):

    def __init__(self, history, name_of_user_id='user_id'):

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id
        
    def predict(self, df):
        correct = float(df['history_correct'])
        total = float(df['history_correct'])+float(df['history_wrong'])
        prob = correct/total
        return prob
    
    def assessment_pass_likelihoods(self, df):
        probs = []
        for i in range(len(df)):
            history = df.iloc[i]
            pred = self.predict(history)
            probs.append(pred)
        
        return probs

class RandomModel(models.SkillModel):

    def __init__(self, history, name_of_user_id='user_id'):

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.name_of_user_id = name_of_user_id
        
    def predict(self):
        prob = random.randrange(0,2)
        
        return prob
    
    def assessment_pass_likelihoods(self, df):
        probs = []
        for i in range(len(df)):
            probs.append(random.randrange(0,2))
            
        return probs



class EFCModel(models.SkillModel):
    """
    Class for memory models that predict recall likelihood using the exponential forgetting curve
    """

    def __init__(self, history, strength_model=None, content_features=None, using_delay=True, 
            using_global_difficulty=True, using_item_bias=True, debug_mode_on=False):

        self.history = history[history['module_type']==datatools.AssessmentInteraction.MODULETYPE]
        self.strength_model = strength_model
       
        self.using_delay = using_delay
        self.using_global_difficulty = using_global_difficulty
        self.using_item_bias = using_item_bias
        self.debug_mode_on = debug_mode_on

        self.idx_of_module_id = {x: i for i, x in enumerate(self.history['module_id'].unique())}
        self.difficulty = None
        
        if content_features is None:
            if self.using_global_difficulty:
                content_features = np.ones((len(self.idx_of_module_id), 1))
        else:    
            content_features = np.array([content_features[module_id] \
                    for module_id in self.history['module_id'].unique()])
            content_features = preprocessing.scale(content_features)
            if self.using_global_difficulty:
                content_features = preprocessing.add_dummy_feature(content_features)
        self.content_features = content_features

        if self.content_features is None and not self.using_item_bias:
            raise ValueError('The log-linear difficulty model has not been defined!')

    def extract_examples(self, df, filter_first_ixns=True):
        """
        Get delays, memory strengths, module indices, and outcomes for a set of interactions

        :param pd.DataFrame df: Interaction log data
        :param bool filter_first_ixns: True if the first interaction in a user-item history should
            be removed, False otherwise. These interactions are marked by tlast = np.nan.

        :rtype: (np.array,np.array,np.array,np.array)
        :return: A tuple of (delays, memory strengths, module indices, outcomes)
        """
    
        if self.using_delay:
            if filter_first_ixns:
                df = df[~np.isnan(df['time_elapsed'])]
            timestamps = np.array(df['timestamp'].values)
            previous_review_timestamps = np.array(df['time_elapsed'].values)
            delays = 1 + (timestamps - previous_review_timestamps) / 86400
        else:
            delays = 1
       
        strengths = 1 if self.strength_model is None else np.array(df[self.strength_model].values)
        module_idxes = np.array(df['module_id'].map(self.idx_of_module_id).values)
        outcomes = np.array(df['outcome'].apply(lambda x: 1 if x else 0).values)
        
        return delays, strengths, module_idxes, outcomes

    def fit(self, learning_rate=0.5, ftol=1e-6, max_iter=1000,
            coeffs_regularization_constant=1e-3, item_bias_regularization_constant=1e-3):
        """
        Learn model hyperparameters using MAP estimation

        Uses batch gradient descent with a fixed learning rate and a fixed threshold on 
            the relative difference between consecutive loss function evaluations 
            as the stopping condition

        Uses the log-linear item difficulty model

        :param float learning_rate: Fixed learning rate for batch gradient descent
        :param float ftol: If the relative difference between consecutive loss function 
            evaluations falls below this threshold, then gradient descent has 'converged'

        :param int max_iter: If the stopping condition hasn't been met after this many iterations, 
            then stop gradient descent

        :param float coeffs_regularization_constant: Coefficient of L2 penalty on coefficients
            in log-linear difficulty model

        :param float item_bias_regularization_constant: Coefficient of L2 penalty on item bias
            term in log-linear difficulty model
        """

        delays, strengths, module_idxes, outcomes = self.extract_examples(self.history)

        eps = 1e-9 # smoothing parameter for likelihoods
        if self.content_features is not None:
            if self.using_item_bias:
                def loss((coeffs, item_biases)):
                    """
                    Compute the average negative log-likelihood and regularization penalty 
                    given the data and hyperparameter values

                    :param np.array coeffs: Coefficientficients of log-linear difficulty model
                    :param float item_bias: Item bias term in log-linear difficulty model

                    :rtype: float
                    :return: Value of loss function evaluated at current parameter values
                    """

                    difficulties = np.exp(-(np.einsum(
                        'i, ji -> j', coeffs, self.content_features[module_idxes, :]) \
                                + item_biases[module_idxes]))
                    pass_likelihoods = np.exp(-difficulties*delays/strengths)
                    log_likelihoods = outcomes*np.log(pass_likelihoods+eps) \
                            + (1-outcomes)*np.log(1-pass_likelihoods+eps)
                    regularizer = coeffs_regularization_constant * np.linalg.norm(coeffs)**2 \
                            + item_bias_regularization_constant * np.linalg.norm(item_biases)**2
                    return -np.mean(log_likelihoods) + regularizer
            else:
                def loss(coeffs):
                    """
                    Compute the average negative log-likelihood and regularization penalty 
                    given the data and hyperparameter values

                    :param np.array coeffs: Coefficients of log-linear difficulty model
                    :param float item_bias: Item bias term in log-linear difficulty model

                    :rtype: float
                    :return: Value of loss function evaluated at current parameter values
                    """

                    difficulties = np.exp(-np.einsum(
                        'i, ji -> j', coeffs, self.content_features[module_idxes, :]))
                    pass_likelihoods = np.exp(-difficulties*delays/strengths)
                    log_likelihoods = outcomes*np.log(pass_likelihoods+eps) \
                            + (1-outcomes)*np.log(1-pass_likelihoods+eps)
                    regularizer = coeffs_regularization_constant * np.linalg.norm(coeffs)**2
                    return -np.mean(log_likelihoods) + regularizer
        else:
            def loss(item_biases):
                """
                Compute the average negative log-likelihood and regularization penalty 
                given the data and hyperparameter values

                :param float item_bias: Item bias term in log-linear difficulty model

                :rtype: float
                :return: Value of loss function evaluated at current parameter values
                """

                difficulties = np.exp(-item_biases[module_idxes])
                pass_likelihoods = np.exp(-difficulties*delays/strengths)
                log_likelihoods = outcomes*np.log(pass_likelihoods+eps) \
                        + (1-outcomes)*np.log(1-pass_likelihoods+eps)
                regularizer = item_bias_regularization_constant * np.linalg.norm(item_biases)**2
                return -np.mean(log_likelihoods) + regularizer

        gradient_fun = grad(loss) # take the gradient of the loss function

        if self.content_features is not None:
            coeffs = np.random.random(self.content_features.shape[1])
        else:
            coeffs = 0

        if self.using_item_bias:
            item_biases = np.random.random(len(self.idx_of_module_id))
        else:
            item_biases = 0
        
        losses = []
        for _ in xrange(max_iter):
            # perform gradient descent update
            if self.content_features is not None:
                if self.using_item_bias:
                    grad_coeffs, grad_item_biases = gradient_fun((coeffs, item_biases))
                    item_biases -= grad_item_biases * learning_rate
                else:
                    grad_coeffs = gradient_fun(coeffs)
                coeffs -= grad_coeffs * learning_rate
            else:
                grad_item_biases = gradient_fun(item_biases)
                item_biases -= grad_item_biases * learning_rate
            
            # evaluate loss function at current difficulty value
            if self.content_features is not None:
                if self.using_item_bias:
                    loss_params = (coeffs, item_biases)
                else:
                    loss_params = coeffs
            else:
                loss_params = item_biases
            losses.append(loss(loss_params))
            
            # evaluate stopping condition
            if len(losses) > 1 and (losses[-2] - losses[-1]) / losses[-2] < ftol:
                break

        self.difficulty = np.exp(-((np.einsum(
            'i, ji -> j', coeffs, self.content_features) \
                    if self.content_features is not None else 0) + item_biases))

        if self.debug_mode_on: 
            # visual check for convergence
            plt.xlabel('Iteration')
            plt.ylabel('Average negative log-likelihood + regularizer')
            plt.plot(losses)
            plt.show()

            # check distribution of learned difficulties
            plt.xlabel(r'Item Difficulty $\theta_i$')
            plt.ylabel('Frequency (Number of Items)')
            plt.hist(self.difficulty)
            plt.show()
    
    
    
    