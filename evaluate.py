from lentil import datatools
from lentil import models
from lentil import evaluate

import numpy as np
import pandas as pd

import tools as t
import predictive_model as m

def train_efc(data, user_id='student_id'):
    model = m.EFCLinear(data, user_id)
    model.fit()
    
    return model

def train_onepl(data, user_id='student_id'):
    model = models.OneParameterLogisticModel(data, select_regularization_constant=True, name_of_user_id=user_id)
    model.fit()
    
    return model
    
def train_logistic(data, user_id='student_id'):
    model = m.LogisticRegressionModel(data, user_id)
    model.fit()
    
    return model
    
efc_model = train_efc(t.spanish_hist.data)
evaluate.training_auc(efc_model, t.filtered_spanish)

onepl_model = train_onepl(t.spanish_hist.data)
evaluate.training_auc(onepl_model, t.filtered_spanish)

logistic_model = train_logistic(t.spanish_hist.data)
evaluate.training_auc(logistic_model, t.filtered_spanish)
evaluate.training_auc(logistic_model, t.filtered_spanish)