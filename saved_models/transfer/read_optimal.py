import sys
import pickle

input_file = sys.argv[1]

pickle_off = open(input_file, 'rb')
print(pickle.load(pickle_off))
