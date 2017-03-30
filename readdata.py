import argparse
import csv
import datetime
import math
import pyirt

class item(object):
	def __init__(self, user_id, item_id, time, outcome):
	    self.user_id = user_id
	    self.item_id = item_id
	    self.time = time
	    self.outcome = outcome

class itemHistory(object):
	def __init__(self, list):
	    self.listOfItems = list

class trainingInst(object):
	def __init__(self, user_id, item_id, last_response, timestamp, time_elapsed, history_seen, history_correct,exponential_sum):
		cur_response = 0
		if last_response == 'CORRECT':
			cur_response = 0.75
		else:
			cur_response = 0.25
		self.user_id = user_id
		self.item_id = item_id
		self.last_response = cur_response
		self.timestamp = timestamp
		self.time_elapsed = time_elapsed
		self.history_seen = history_seen
		self.history_correct = history_correct
		self.exponential_sum = exponential_sum

def readData(fname):
	file = open(fname, 'r')
	items = csv.reader(file)
	raw_item_list = list(items)
	item_list = []
	for i in range(1,len(raw_item_list)):
		item_list.append(item(raw_item_list[i][0], raw_item_list[i][3], raw_item_list[i][2], raw_item_list[i][4]))
	
	ultimate = []
	interactions = []
	for x in range(len(item_list)):
		#temp vars
		temp = []
		unique_combo = True
		unique_seq = True

		user_id = item_list[x].user_id
		item_id = item_list[x].item_id

		#check if combo has not already existed
		if len(interactions) > 0:
			for c in range(0, len(interactions)):
				#print [[user_id, item_id], interactions[c]]
				if [user_id, item_id] == interactions[c] and unique_combo == True:
					unique_combo = False
			if unique_combo == True:
				interactions.append([user_id, item_id])	
				temp.append(item_list[x])

		else:
			interactions.append([user_id, item_id])	
			temp.append(item_list[x])

		flag = 0

		if unique_combo == True:
			for y in range(0, len(item_list)):
				new_user_id = item_list[y].user_id
				new_item_id = item_list[y].item_id

				
				if (new_user_id == user_id and new_item_id == item_id):
					if flag == 1:
						temp.append(item_list[y])
					else:
						flag = 1

			ultimate.append(itemHistory(temp))

	return ultimate
def convertHistoryToList(itemHistory):
	listOfItems = itemHistory.listOfItems
	display = ''
	display = display + listOfItems[0].user_id 
	display = display + ',' + listOfItems[0].item_id
	for i in range(len(listOfItems)):
		time_elapsed = str(convertTimeToTimestamp(listOfItems[i].time)-convertTimeToTimestamp(listOfItems[i-1].time))

		if i > 0:
			display = display + ',' + listOfItems[i].outcome + ':' + time_elapsed
		else: 
			display = display + ',' + listOfItems[i].outcome + ': First Attempt'

	return display

def convertTimeToTimestamp(input_time):
	yearly = input_time.split('/')[0].split('-')
	daily = input_time.split('/')[1].split(':')
	dt = datetime.datetime(int(yearly[0]), int(yearly[1]), int(yearly[2]), int(daily[0]), int(daily[1]), int(daily[2]))
	unix = int(dt.strftime('%s'))
	return unix


def outputHistories(input_file, moreThan=0, writeFile=True, user_id=None, item_id=None):
	histories = readData(input_file)
	file = open('histories.txt','w') 
	for i in range(len(histories)):
		if (user_id != None):
			if (len(histories[i].listOfItems) > moreThan and histories[i].listOfItems[0].user_id == user_id):
				if writeFile:
					file.write(convertHistoryToList(histories[i]))
				else:
					print convertHistoryToList(histories[i])
		else:
			if (len(histories[i].listOfItems) > moreThan):
				if writeFile:
					file.write(convertHistoryToList(histories[i]))
				else:
					print convertHistoryToList(histories[i])

def outputTrainingInstances(input_file):
	histories = readData(input_file)
	instances = []
	for i in range(len(histories)):
		current_history = histories[i].listOfItems
		history_correct = 0
		sequence = []
		expo = 0.0
		expo_incre = 1.0
		#need to fix first two responses in history correct
		for x in range(0, len(current_history)):
			user_id = current_history[x].user_id
			item_id = current_history[x].item_id
			last_response = current_history[x].outcome
			timestamp = str(convertTimeToTimestamp(current_history[x].time))
			time_elapsed = str(convertTimeToTimestamp(current_history[x].time)-convertTimeToTimestamp(current_history[x-1].time))		
			history_seen = x
			#create instance
			instance = trainingInst(user_id, item_id, last_response, timestamp, time_elapsed, history_seen, history_correct, expo)
			
			#increments vars
			if last_response == 'CORRECT':
				history_correct = history_correct + 1
				sequence.append(1)
			else:
				sequence.append(0)
			if x>=2:
				instances.append(instance)

			#calculate expo
			expo = 0.0
			for y in range(len(sequence)):
				expo_incre = math.pow(0.8,y)
				if list(reversed(sequence))[y] == 1.0:
					expo += expo_incre


	return instances

def instancesToFile(list):
	file = open('data.txt', 'w')
	file.write('p_recall,timestamp,time_elapsed,user_id,item_id,history_seen,history_correct,exponential')
	for i in range(len(list)):
		file.write(str(list[i].last_response) + ',' + str(list[i].timestamp) + ',' + str(list[i].time_elapsed) + ',' +  list[i].user_id + ',' + list[i].item_id + ',' + str(list[i].history_seen) + ',' + str(list[i].history_correct) +',' + str(list[i].exponential_sum))


#arguments
argparser = argparse.ArgumentParser(description='Convert student data into a list of item histories')
argparser.add_argument('input_file', action="store", help='student data for reading')
args = argparser.parse_args()

outputHistories(args.input_file, 3, True)
instancesToFile(outputTrainingInstances(args.input_file))



