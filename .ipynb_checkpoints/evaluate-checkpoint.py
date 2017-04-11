from lentil import datatools
from lentil import models
from lentil import evaluate

import numpy as np
import pandas as pd

import tools as t
import predictive_model as m
import math
import constants
from sklearn.metrics import roc_auc_score

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
    
def online_prediction_acc(model, all_data, train_data, test_data):
    #for auc
    preds = []
    y = []
    
    #data
    test_students = test_data._student_idx
    test_modules = test_data._assessment_idx
    train_modules = train_data._assessment_idx
    all_students = all_data._student_idx
    all_modules = all_data._assessment_idx
    
    #functions
    def getCurrentStudent(index, students):
        return students.keys()[students.values().index(index)]

    def getModulesFromStudents(student_id):
        idx = all_students.keys()[all_students.values().index(student_id)]
        a = list(all_data.data[all_data.data['student_id'] == idx]['module_id'].drop_duplicates())
        a = [int(i) for i in a]
        a.sort()

        return a
        
    #accuracy
    correct = 0
    wrong = 0
    
    if model == 'irt':
        #IRT evaluation
        onepl_model = train_onepl(train_data.data, train_data.data)
        initial_difficulties = getIRTParameters(onepl_model)[1]

        for i in range(test_data.num_students()-1):
            current_student_history = test_data.data[test_data.data['student_id'] == getCurrentStudent(i,test_students)]
            print 'current student: ' + getCurrentStudent(i,test_students) + ', with ' + str(len(current_student_history)) +' interactions.'
            if i > 0:
                print float(correct)/(float(correct)+float(wrong))

            for x in range(len(current_student_history)):
                #predict (return prob)
                prob = 0
                outcome = current_student_history.iloc[x]['outcome']
                if x < 2:
                    prob = 0
                else:
                    df = pd.concat([train_data.data, current_student_history[0:x]])
                    onepl_model = train_onepl(None, df)  
                    try:
                        difficulty = initial_difficulties[train_modules[current_student_history.iloc[x]['module_id']]]
                    except:
                        difficulty = 0.3

                    student_ids = onepl_model.idx_of_student_id
                    ability = getIRTParameters(onepl_model)[0][student_ids[getCurrentStudent(i,test_students)]]
                    #print ability, difficulty
                    prob = 1 / (1 + math.exp(-(ability + difficulty)))
                
                preds.append(prob)
                y.append(int(outcome))
                if round(prob) == outcome:
                    correct += 1
                else: 
                    wrong += 1  
    else:
        #Logistic/EFC evaluation
        if model == 'efc':
            clf = train_efc(train_data.data, train_data.data)  
        elif model == 'logistic':
            clf = train_logistic(train_data.data, train_data.data)

        for i in range(test_data.num_students()-1):
            current_student_history = test_data.data[test_data.data['student_id'] == getCurrentStudent(i,test_students)]
            #print 'current student: ' + getCurrentStudent(i,test_students) + ', with ' + str(len(current_student_history)) +' interactions.'
            #if i > 0:
                #print float(correct)/(float(correct)+float(wrong))

            for x in range(len(current_student_history)):
                #predict (return prob)
                outcome = current_student_history.iloc[x]['outcome']
                time_elapsed = current_student_history.iloc[x]['time_elapsed']
                fv = []
                for c in range(len(constants.FEATURE_NAMES)):
                    feature = current_student_history.iloc[x][constants.FEATURE_NAMES[c]]
                    if m.isfloat(feature):
                        fv.append(float(feature))
                    elif m.isint(feature):
                        fv.append(int(feature))
                        
                if model == 'efc':
                    prob = clf.predict(clf.clf, fv, time_elapsed)
                elif model == 'logistic': 
                    prob = clf.predict(fv)
                
                #auc
                preds.append(prob)
                y.append(int(outcome))
                
                #evaluate
                if round(prob) == outcome:
                    correct += 1
                else: 
                    wrong += 1 
                    
    print 'ACC: ' + str(float(correct)/(float(correct)+float(wrong))) + ', with ' + str(correct) + ' correct and ' + str(wrong) + 'wrong.'
    
    print 'AUC: ' + str(roc_auc_score(y, preds))
   
    return float(correct)/(float(correct)+float(wrong)), correct, wrong

        
        
        
        