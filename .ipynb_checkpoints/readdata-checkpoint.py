import argparse
import csv
import datetime
import math
import codecs
import pickle
import constants
import itertools

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
    def __init__(self, user_id, item_id, last_response, timestamp, time_elapsed, features, correct='CORRECT'):
        cur_response = False
        if last_response == correct:
            cur_response = True
        else:
            cur_response = False
        self.user_id = user_id
        self.item_id = item_id
        self.last_response = cur_response
        self.timestamp = timestamp
        self.time_elapsed = time_elapsed
        self.features = features

def changeToInt(text):
    if text == 'CORRECT':
        return 1
    else:
        return 0

def readData(fname, user, module, timestamp, outcome):
    column_index = {'user_id':user, 'module_id':module, 'outcome': outcome, 'timestamp': timestamp}

    file = open(fname, 'r')
    items = csv.reader(file)
    raw_item_list = list(items)
    item_list = []
    user_id_table = {}
    item_id_table = {}
    count_user = 0
    count_item = 0

    print 'reading data'
    for i in range(1, len(raw_item_list)):
        #print str(i) + ' out of ' + str(len(raw_item_list))
        try:
            #if there is such id
            uid = user_id_table[raw_item_list[i][column_index['user_id']]]
        except :
            #if no such id, then create one
            count_user += 1
            user_id_table[str(raw_item_list[i][column_index['user_id']])] = count_user
        try :
            eid = item_id_table[raw_item_list[i][column_index['module_id']]]
        except:
            count_item += 1
            item_id_table[str(raw_item_list[i][column_index['module_id']])] = count_item

        raw_item_list[i][column_index['user_id']] = user_id_table[str(raw_item_list[i][column_index['user_id']])]
        raw_item_list[i][column_index['module_id']] = item_id_table[str(raw_item_list[i][column_index['module_id']])]
    print 'finished initial process'
    for i in range(1,len(raw_item_list)):
        #item list is an array of ALL the interaction items
        item_list.append(item(raw_item_list[i][column_index['user_id']], raw_item_list[i][column_index['module_id']],raw_item_list[i][column_index['timestamp']], raw_item_list[i][column_index['outcome']]))

    #final array of all item HISTORIES
    ultimate = []
    #existing combinations
    interactions = []
    for x in range(len(item_list)):
        #print str(x) + ' out of ' + str(len(item_list))
        #temp vars
        temp = []
        unique_combo = True
        unique_seq = True

        user_id = item_list[x].user_id
        item_id = item_list[x].item_id

        #if there has been existing combinations
        if len(interactions) > 0:
            #check if the current item's interaction is unique
            for c in range(0, len(interactions)):
                if [user_id, item_id] == interactions[c] and unique_combo == True:
                    unique_combo = False

            if unique_combo == True:
                interactions.append([user_id, item_id])
                temp.append(item_list[x])

        else:
            interactions.append([user_id, item_id])
            temp.append(item_list[x])

        #makes sure first interaction not added (it is already added)
        flag = 0

        #search and add the other interactions excluding the first one
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

    with open('history.pkl', 'wb') as handle:
        pickle.dump(ultimate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ultimate

def convertHistoryToList(itemHistory):
    listOfItems = itemHistory.listOfItems
    display = ''
    display = display + str(listOfItems[0].user_id)
    display = display + ',' + str(listOfItems[0].item_id)
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
    histories = readData(input_file,6,0,1,2)
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
#activation value
def activationValue(old):
    x = old[1:len(old)]
    output_act = []
    output_delays = [0.177]
    # x is a list
    for item in range(0, len(x)):
        recursive = []
        if item == 0:
            act = math.log(x[item]**-0.177)
            output_act.append(act)
        else:
            delay = (0.217*math.exp(1)**output_act[item-1])+0.177
            output_delays.append(delay)
            for time in range(0, len(output_delays)):
                if time == 0:
                    recursive.append(x[len(output_delays)-1]**-(output_delays[time])) 
                else:
                    recursive.append((x[len(output_delays)-1] - (x[len(output_delays) - (time+1)]))**-(output_delays[time]))  

            act = math.log(sum(recursive))
            output_act.append(act)
    return output_act[-1]
                        
def outputTrainingInstances(input_file, user, module, time, outcome, ts, pickled=None, correct='CORRECT'):
    if pickled == None:
        histories = readData(input_file, user, module, time,outcome)
    else:
        with open(pickled, 'rb') as handle:
            histories = pickle.load(handle)

    instances = []
    print 'getting features'
    for i in range(len(histories)):
        #print str(i) + ' out of ' + str(len(histories))

        #general
        current_history = histories[i].listOfItems
        sequence = []
        time_sequence = []
        activation_sequence = []

        #features
        expo = 0.0
        history_correct = 0
        history_wrong = 0
        longest_wrong_streak = 0
        longest_right_streak = 0
        average_outcome = 0.5
        average_time = 0
        activation = -0.63

        #other
        expo_incre = 1.0

        for x in range(0, len(current_history)):
            features = [history_correct, history_wrong, expo, longest_wrong_streak, longest_right_streak, average_outcome, average_time, activation]
            
            if current_history[x].time != 'US/Eastern':
                user_id = current_history[x].user_id
                item_id = current_history[x].item_id
                last_response = current_history[x].outcome
                if ts == False:
                    timestamp = str(convertTimeToTimestamp(current_history[x].time))
                    time_elapsed = str(convertTimeToTimestamp(current_history[x].time)-convertTimeToTimestamp(current_history[x-1].time))
                else:
                    timestamp = current_history[x].time
                    time_elapsed = current_history[x].time - current_history[x-1].time
                    
        #activation
                if x>= 1:   
                    time_sequence.append(time_elapsed)
                if len(time_sequence) >= 1:
                    activation_sequence.append(sum([int(i) for i in time_sequence]))  
                    activation = activationValue(activation_sequence)
                else:
                    activation_sequence.append(0)
                    
        #create instance
                instance = trainingInst(user_id, item_id, last_response, timestamp, time_elapsed, features)
                if x >= 1:
                    instances.append(instance)
                
            
        #FEATURES!

        #history correct
                if last_response == correct:
                    history_correct += 1
                    sequence.append(1)
        #history wrong
                else:
                    history_wrong += 1
                    sequence.append(0) 
                
                
        #calculate expo
                expo = 0.0
                for y in range(len(sequence)):
                    expo_incre = math.pow(0.8,y)
                    if list(reversed(sequence))[y] == 1:
                        expo += expo_incre
        #streaks               
                streaks = [(k, sum(1 for i in g)) for k,g in itertools.groupby(sequence)]
            
        #calculate longest wrong streak
                try:
                    longest_wrong_streak = max([t for s,t in streaks if s == 0])
                except:
                    longest_wrong_streak = 0
        #calculate longest correct streak
                try: 
                    longest_right_streak = max([t for s,t in streaks if s == 1])
                except:
                    longest_right_streak = 0
                    
        #average outcome
                float_sequence = [float(i) for i in sequence]
                average_outcome = sum(float_sequence)/float(len(float_sequence))
                
        #average time_elapsed
                int_time_sequence = [int(i) for i in time_sequence]
                if x == 0:
                    average_time = 0
                else:
                    average_time = float(sum(int_time_sequence))/float(len(int_time_sequence))

    return instances

def instancesToFile(list, fname):
    print 'writing data...'
    file = open(fname, 'w')
    file.write('outcome,timestamp,time_elapsed,student_id,module_id,module_type,' + ','.join(constants.FEATURE_NAMES) + '\n')
    for i in range(len(list)):
        feature_string = ','.join([str(x) for x in list[i].features])
        file.write(str(list[i].last_response) + ',' + str(list[i].timestamp) + ','  + str(list[i].time_elapsed) + ',' +  str(list[i].user_id) + ',' + str(list[i].item_id) + ',assessment,' + feature_string + '\n')

def getTrainingInstances(file, fname, user, module, time, outcome):
    instancesToFile(outputTrainingInstances(file, user, module, time, outcome, False), fname)

#arguments
# argparser = argparse.ArgumentParser(description='Convert student data into a list of item histories')
# argparser.add_argument('input_file', action="store", help='student data for reading')
# argparser.add_argument('-user', action="store", dest="user_index", type=int, default=None)
# argparser.add_argument('-module', action="store", dest="module_index", type=int, default=None)
# argparser.add_argument('-time', action="store", dest="time_index", type=int, default=None)
# argparser.add_argument('-outcome', action="store", dest="outcome_index", type=int, default=None)
# argparser.add_argument('-t', action="store_true", default=False, help='already have timestamp')
# argparser.add_argument('-correct', action="store", dest="correct_str", type=str, default='TRUE')

# args = argparser.parse_args()

#command line
#instancesToFile(outputTrainingInstances(args.input_file, args.user_index, args.module_index, args.time_index, args.outcome_index, args.t), 'data.txt')
