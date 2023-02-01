# +-------------------------+-----------------------------+
# Written By   : Akpokiro Victor
# +-------------------------+-----------------------------+
# Filename     : test.py
# +-------------------------+-----------------------------+
# Description  : CNNSplice: Robust Models for Splice Site 
#					Prediction Using 
#					Deep Convolutional Neural Networks. 
#		To test CNNSplice model
# +-------------------------+-----------------------------+
# Company Name :  OluwadareLab UCCS
# +-------------------------+-----------------------------+
# This also contain piece of code from:
# Wang, R et al., (2019) SpliceFinder source code [Source code]. 
# https://github.com/deepomicslab/SpliceFinder/blob/master/SpliceFinder_sourcecode/CNN_model.py
# +-------------------------+-----------------------------+



from __future__ import print_function
import numpy as np
import time
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import *
from sklearn import tree, metrics
from sklearn.metrics import precision_score, recall_score, classification_report, roc_auc_score
import argparse




def load_data(mode, datatype, seq):
	y_test = np.loadtxt(f'./data/{mode}/{datatype}/test_{seq}_{datatype}_lbl')
	x_test = np.loadtxt(f'./data/{mode}/{datatype}/test_{seq}_{datatype}')

	x_test = x_test.reshape(-1,400, 4)
	y_true = y_test
	y_test = to_categorical(y_test, num_classes=2)

	return x_test, y_test


def testing_process(x_test, y_test, seq, seq_name, name, datatype =""):

	model = load_model(f'./models/{seq_name}_cnnsplice_{datatype}.h5')
	start_time = time.time()
	print(model.summary())
	loss,accuracy = model.evaluate(x_test,y_test)
	print('testing accuracy: {}_{}_{}_{}'.format(seq, name, datatype, accuracy))

	prob = model.predict(x_test)
	predict = model.predict(x_test)
	predict = to_categorical(predict, num_classes=2)
	y_true = y_test


	predicted = np.argmax(prob, axis=1)
	report = classification_report(np.argmax(y_true, axis=1), predicted, output_dict=True )

	macro_precision =  report['macro avg']['precision'] 
	macro_recall = report['macro avg']['recall']    
	macro_f1 = report['macro avg']['f1-score']
	class_accuracy = report['accuracy']

	data_metrics = {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1, "class_accuracy": class_accuracy, "accuracy": accuracy}
	print(data_metrics)

	with open(f'./log/test_logfile_metrics_{datatype}.txt', 'w') as fl:
		fl.write(str(data_metrics))



def main(name, mode):

	seq = "acc"
	seq_name = "acceptor"
	list_name = ["hs", "at", "oriza", "d_mel", "c_elegans"]

	for datatype in list_name:
		x_test,y_test = load_data(mode, datatype, seq_name)
		testing_process(x_test,y_test, seq, seq_name, name, datatype=datatype)

	print("======================")
	print("======================")
	print("======================")
	print("======================")
	print("======================")
	print('Start Donor Convolution')

	seq = "don"
	seq_name = "donor"
	for datatype in list_name:
		x_test,y_test = load_data(mode, datatype, seq_name)
		testing_process(x_test,y_test, seq, seq_name, name, datatype=datatype)




def app_init():

	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--name", type=str, required=True, help="name of convolutional model")
	parser.add_argument("-m", "--mode", type=str, required=True, help="balanced or imbalanced")


	args = parser.parse_args()
	name = args.name
	mode = args.mode


	main(name, mode)




if __name__ == '__main__':
	app_init()



