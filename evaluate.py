from lentil import datatools
from lentil import models
from lentil import evaluate

import numpy as np
import pandas as pd

import tools as t
import predictive_model as m
import math

#Train Models
def train_efc(history, filtered_history, split_history=None):
    model = m.EFCLinear(filtered_history, name_of_user_id='student_id')
    model.fit()
    
    return model

def train_onepl(history, filtered_history, split_history=None):
    model = models.OneParameterLogisticModel(filtered_history, select_regularization_constant=True, name_of_user_id='student_id')
    model.fit()
    
    return model
    
def train_twopl(history, filtered_history, split_history=None):
    model = models.TwoParameterLogisticModel(filtered_history, select_regularization_constant=True, name_of_user_id='student_id')
    model.fit()
    
    return model
    
def train_logistic(history, filtered_history, split_history=None):
    model = m.LogisticRegressionModel(filtered_history, name_of_user_id='student_id')
    model.fit()
    
    return model

def trainAll(history):
    return train_efc(history, history), train_onepl(history, history), train_twopl(history, history), train_logistic(history, history)

#OnePL Model
def getIRTParameters(model, p=False):
    difficulties =  model.model.coef_[0, model.num_students:]
    abilities = model.model.coef_[0, :model.num_students]
    if p:
        print str(model.num_students) + ' students; ' + str(model.num_assessments) + ' assessments'
        print str(len(abilities))  + ' abilities; ' + str(len(difficulties)) + ' difficulties'
    return abilities, difficulties

def getIRTProb(model, df, index):
    ability = getIRTParameters(model)[0][int(df.iloc[index]['student_id'])-1]
    difficulty = getIRTParameters(model)[1][int(df.iloc[index]['module_id'])-1]
    prob = sigmoid(ability - difficulty)
    return prob
    
def getIRTProbs(model, df):
    probs = []
    for index in range(len(df)):
        try:
            ability = getIRTParameters(model)[0][int(df.iloc[index]['student_id'])-1]
            difficulty = getIRTParameters(model)[1][int(df.iloc[index]['module_id'])-1]
            prob = sigmoid(ability + difficulty)
        except:
            prob = 2
        probs.append(prob)
    return probs
    
def makeIRTDf(user_id, module_id):
    df = pd.DataFrame(data = [[user_id, module_id]], columns = ['student_id', 'module_id'])
    return df
    
#Evaluate Functions

def getResults(data, num_folds=10, random_truncations=True):
    model_builders = {
        '1PL IRT' : train_onepl,
        'EFC' : train_efc,
        'LR' : train_logistic
    }
    results = evaluate.cross_validated_auc(model_builders,data,num_folds=num_folds,random_truncations=random_truncations)
    return results
    
def getACC(model, data, onepl=False):
    correct = 0
    wrong = 0
    y = list(data['outcome'])
    print type(model)
    if onepl:
        x = getIRTProbs(model, data)
    else:
        x = model.assessment_pass_likelihoods(data)
    for i in range(len(y)):
        prediction = x[i]
        if round(prediction) == y[i]:
            correct += 1
        elif round(prediction) == 2:
            print 'lol'
        else:
            wrong += 1
    total = float(correct) + float(wrong)
    return float(correct)/total

def getTrainingAUCs(models, history):
    aucs = []
    for i in range(len(models)):
        aucs.append(evaluate.training_auc(models[i], history))
    return aucs

def overallAccuracy(model_names, results):
    training_aucs = []
    validation_aucs = []
    
    for i in range(len(model_names)):
        training_aucs.append(results.training_auc_mean(model_names[i]))
        validation_aucs.append(results.validation_auc_mean(model_names[i]))
    
    training_string = ''
    validation_string = ''
    
    for x in range(len(model_names)):
        training_string = training_string + ' ' + model_names[x] + ': ' + str(training_aucs[x])
        validation_string = validation_string + ' ' + model_names[x] + ': ' + str(validation_aucs[x])
    
    print training_string
    print validation_string
        
        
        
        
        