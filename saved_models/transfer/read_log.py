import pickle
import sys
from pprint import pprint

#file_name = 'training_4_10/transfer_log_' + str(sys.argv[1])
#objs = []
#file_name = str(sys.argv[1])
folder = str(sys.argv[1])
d = {}
val = {}
for num in range(10, 20):
    file_name = folder + '/transfer_log_' + folder + '_' + str(num)
    pickle_off = open(file_name, "rb")
    emp = pickle.load(pickle_off)
    d[num] = emp['val_categorical_accuracy']
    val[num] = emp['validation_majority_acc'][0]

pprint(d)
pprint(val)
