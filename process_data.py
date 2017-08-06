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

rm.getTrainingInstances('raw_data/' + args.raw_data,'processed_data/' + args.processed_data, 0, 2, 1, 3, ts=False, correct='CORRECT')

data = t.textToInteractionHistory('processed_data/' + args.processed_data, 'timestamp', 'student_id','module_id','outcome', 'True')

filtered = t.filterHistory(data.data)

t.savePickle(filtered, 'datasets/' + args.data_name + '.pkl')

argparser = argparse.ArgumentParser(description='Convert student data into a list of item histories, and generate features. ')
argparser.add_argument('raw_data', action="store", help='student data for reading')
argparser.add_argument('processed_data', action="store", help='processed data')
argparser.add_argument('data_name', action="store", help='name of dataset')


args = argparser.parse_args()