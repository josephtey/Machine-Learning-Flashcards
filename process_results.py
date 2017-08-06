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

data = t.loadPickle('datasets/' + args.data)
results = e.getResults(data, 4, True, 0.2)

t.savePickle(results, 'results/' + args.output)

argparser = argparse.ArgumentParser(description='Get results from pickled dataset.')
argparser.add_argument('data', action="store", help='pickled dataset')
argparser.add_argument('output', action="store", help='output dataset')

args = argparser.parse_args()