# +-------------------------+-----------------------------+
# Written By   : Akpokiro Victor
# +-------------------------+-----------------------------+
# Filename     : test.py
# +-------------------------+-----------------------------+
# Description  : CNNSplice: Robust Models for Splice Site 
#					Prediction Using 
#					Deep Convolutional Neural Networks. 
# +-------------------------+-----------------------------+
# Company Name :  OluwadareLab UCCS
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




def load_data(datatype, seq):
	y_test = np.loadtxt(f'./processed_data/{datatype}/test_{seq}_{datatype}_lbl')
	x_test = np.loadtxt(f'./processed_data/{datatype}/test_{seq}_{datatype}')

	x_test = x_test.reshape(-1,400, 4)
	y_true = y_test
	y_test = to_categorical(y_test, num_classes=2)

	return x_test, y_test


def testing_process(x_test, y_test, seq, name, datatype =""):

	model = load_model(f'./models/CNN_{seq}_{name}_{datatype}.h5')
	start_time = time.time()
	print(model.summary())
	loss,accuracy = model.evaluate(x_test,y_test)
	print('testing accuracy: {}_{}_{}_{}'.format(seq, name, datatype, accuracy))

	prob = model.predict(x_test)
	predict = model.predict(x_test)
	predict = to_categorical(predict, num_classes=2)
	y_true = y_test

	auc = roc_auc_score(y_true, model.predict_proba(x_test), multi_class='ovr')

	predicted = np.argmax(prob, axis=1)
	report = classification_report(np.argmax(y_true, axis=1), predicted, output_dict=True )

	macro_precision =  report['macro avg']['precision'] 
	macro_recall = report['macro avg']['recall']    
	macro_f1 = report['macro avg']['f1-score']
	class_accuracy = report['accuracy']

	data_metrics = {"auc_score": auc, "precision": macro_precision, "recall": macro_recall, "f1": macro_f1, "class_accuracy": class_accuracy, "accuracy": accuracy}
	print(data_metrics)

	with open(f'./logs/ltest_file_metrics_{datatype}.txt', 'w') as fl:
		fl.write(str(data_metrics))



def main(name):

	seq = "acc"
	seq_name = "acceptor"
	list_name = ["hs", "at", "oriza", "d_mel", "c_elegans"]

	for datatype in list_name:
		x_test,y_test = load_data(datatype, seq_name)
		testing_process(x_test,y_test, seq, name, datatype=datatype)

	print("======================")
	print("======================")
	print("======================")
	print("======================")
	print("======================")
	print('Start Donor Convolution')

	seq = "don"
	seq_name = "donor"
	for datatype in list_name:
		x_test,y_test = load_data(datatype, seq_name)
		testing_process(x_test,y_test, seq, name, datatype=datatype)




def app_init():

	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser()
	parser.add_argument("-n", "--name", type=str, required=True, help="name of convolutional model")
	# parser.add_argument("-m", "--mode", type=str, required=True, help="balanced or imbalanced")
	# parser.add_argument("-o", "--organism", type=str, required=True, help="dataset organism")
	parser.add_argument("-g", "--encoded_seq", str=str, metavar='FILE', required=False, help="one-hot encoded genome sequence data file")
	parser.add_argument("-l", "--label", str=str, metavar='FILE', required=False, help="encoded label data")


	args = parser.parse_args()
	name = args.name
	if args.encoded_seq:
		file_encoded_seq = args.encoded_seq
	if args.file_label: 
		file_label = args.label

	main(name)




if __name__ == '__main__':
	app_init()



