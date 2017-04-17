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
from autograd import grad

import plotly

import plotly.plotly as py
import plotly.graph_objs as go 

plotly.tools.set_credentials_file(username='tisijoe', api_key='kv9QRxjplURljsrq6ppg')



#Train Models
def train_efc(history, filtered_history, split_history=None):
    model = m.EFCLinear(filtered_history, name_of_user_id='student_id',omit_strength=False)
    model.fit()
    
    return model


def train_efc_2(history, filtered_history, split_history=None):
    model = m.EFCModel(
        filtered_history, strength_model='history_correct', using_delay=True, 
        using_global_difficulty=True, debug_mode_on=True,
        content_features=None, using_item_bias=True)
    model.fit(
        learning_rate=0.1, 
        #learning_rate=(1 if not using_global_difficulty else 0.1), 
        ftol=1e-6, max_iter=10000,
        coeffs_regularization_constant=1e-6, 
        item_bias_regularization_constant=1e-6)
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

def train_percentage(history, filtered_history, split_history=None):
    model = m.PercentageModel(filtered_history, name_of_user_id='student_id')
    
    return model

def train_random(history, filtered_history, split_history=None):
    model = m.RandomModel(filtered_history, name_of_user_id='student_id')
    
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

def getResults(data, num_folds=10, random_truncations=True, test_p=0.2):
    model_builders = {
        'EFC' : train_efc,
        'LR' : train_logistic,
        'IRT': train_onepl,
        'PERC': train_percentage,
        'RAND': train_random
    }
    results = evaluate.cross_validated_auc(model_builders,data,num_folds=num_folds,random_truncations=random_truncations,size_of_test_set=test_p)
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

def overallAccuracy(model_names, results, type, dataset, plot_boxes = False):
    training_aucs = []
    validation_aucs = []
    test_aucs = []
    test_accs = []
    validation_error = []
    
    
    for i in range(len(model_names)):
        training_aucs.append(results.training_auc_mean(model_names[i]))
        validation_aucs.append(results.validation_auc_mean(model_names[i]))
        test_aucs.append(results.test_auc(model_names[i]))
        test_accs.append(results.test_acc(model_names[i]))
        validation_error.append(results.validation_auc_stderr(model_names[i]))

    #select which property to show
    if type == 'AUC':
        data = test_aucs
    elif type == 'ACC':
        data = test_accs
    elif type == "ERROR":
        data = validation_error
        
    low = min(data)-0.02
    high = max(data)+0.02
    
    data = [go.Bar(
            x=model_names,
            y=data,
            marker = dict(color=['rgba(222,45,38,0.8)', 'rgba(204,204,204,1)','rgba(204,204,204,1)', 'rgba(204,204,204,1)'])
    )]
    
    layout = {
            'xaxis': {'title': 'Models'},
            'yaxis': {'title': type, 'range': [low, high]},
            'barmode': 'relative',
            'title': 'Bar Graph: ' + type + ' of different models on ' + dataset + ' dataset.'
    };

    plotly.offline.plot({'data': data, 'layout': layout}, filename='basic-bar.html')
    
    traces = []
    for i in range(len(model_names)):
        if model_names[i] != 'PERC':
            trace = go.Box(
                y=[training_aucs[i], validation_aucs[i]],
                name=model_names[i]
            )
            traces.append(trace)
    
    layout = {
            'xaxis': {'title': 'Models'},
            'yaxis': {'title': 'AUC', 'range': [0.4, 0.9]},
            'title': 'Validation + Training AUCs'
    };
    
    data = traces
    
    if plot_boxes:
        plotly.offline.plot({'data': data, 'layout': layout})

        
    
def online_prediction_acc(model, all_data, train_data, test_data, trained=None):
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
                print roc_auc_score(y, preds)

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

        for i in range(test_data.num_students()-1):
            current_student_history = test_data.data[test_data.data['student_id'] == getCurrentStudent(i,test_students)]
            
            #if i > 0:
                #print roc_auc_score(y, preds)

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
                    prob = trained.predict(trained.clf, fv, time_elapsed)
                elif model == 'logistic': 
                    fv.append(time_elapsed) 
                elif model == 'percentage':
                    prob = trained.predict(current_student_history.iloc[x])
                elif model == 'random':
                    prob = trained.predict()
                
                #auc
                preds.append(round(prob))
                y.append(int(outcome))
                
                #evaluate
                if round(prob) == outcome:
                    correct += 1
                else: 
                    wrong += 1 
                    
    print sum(preds)/float(len(preds))
    print 'ACC: ' + str(float(correct)/(float(correct)+float(wrong))) + ', with ' + str(correct) + ' correct and ' + str(wrong) + 'wrong.'
    
    print 'AUC: ' + str(roc_auc_score(y, preds))
   
    return float(correct)/(float(correct)+float(wrong)), correct, wrong

        
        
        
        