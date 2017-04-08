from lentil import datatools
from lentil import models
from lentil import evaluate

import numpy as np
import pandas as pd

import tools as t
import predictive_model as m

def train_efc(history, filtered_history, split_history=None):
    model = m.EFCLinear(filtered_history, name_of_user_id='student_id')
    model.fit()
    
    return model

def train_onepl(history, filtered_history, split_history=None):
    model = models.OneParameterLogisticModel(filtered_history, select_regularization_constant=True, name_of_user_id='student_id')
    model.fit()
    
    return model
    
def train_logistic(history, filtered_history, split_history=None):
    model = m.LogisticRegressionModel(filtered_history, name_of_user_id='student_id')
    model.fit()
    
    return model
    
efc_model = train_efc(t.spanish_hist.data)
onepl_model = train_onepl(t.spanish_hist.data)
logistic_model = train_logistic(t.spanish_hist.data)

model_builders = {
    '1PL IRT' : train_onepl,
    'EFC' : train_efc,
    'LR' : train_logistic
}

results = evaluate.cross_validated_auc(
    model_builders,
    t.filtered_spanish,
    num_folds=10,
    random_truncations=True)
