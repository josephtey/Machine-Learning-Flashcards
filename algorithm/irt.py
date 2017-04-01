from pyirt import *
import csv
import argparse

argparser = argparse.ArgumentParser(description='Generate IRT values from dataset.')
argparser.add_argument('input_file', action="store", help='dataset')
args = argparser.parse_args()

file = open(args.input_file, 'r')
src_fp = [[x for x in rec] for rec in csv.reader(file, delimiter=',')]

items = []

def changeToInt(text):
	if text == 'CORRECT':
		return 1
	else: 
		return 0

user_id_table = {}
item_id_table = {}
items = []
count_user = 0
count_item = 0
for i in range(len(src_fp)):
	try:
		#if there is such id
		uid = user_id_table[src_fp[i][0]]
	except :
		#if no such id, then create one
		count_user += 1
		user_id_table[str(src_fp[i][0])] = count_user
	try :
		eid = item_id_table[src_fp[i][3]]
	except:
		count_item += 1
		item_id_table[str(src_fp[i][3])] = count_item
	
	items.append([user_id_table[src_fp[i][0]], item_id_table[src_fp[i][3]], changeToInt(src_fp[i][4])])

# (1)Run by default
item_param, user_param = irt(items)

print user_param