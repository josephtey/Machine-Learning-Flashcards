import tools as t
import readdata as rm
import predictive_model as m
import evaluate as e

from lentil import models
from lentil import evaluate
from lentil import datatools

import pandas as pd
import numpy as np
import pickle
import math
import constants

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

def generate(dataset, output):
	print 'starting ' + output
	history = t.loadPickle(dataset)
	train, test = t.splitHistory(history, 70)
	total, train, test = history, train, test
	for i in range(1):
		print 'starting ' + output + ' ' + str(i)
		results = e.getResults(total, 5, True)
		t.savePickle(results, 'results/secondattempt/' + output + '_' + str(i) + '.pkl')


generate('datasets/chinese.pkl', 'chinese_2006')