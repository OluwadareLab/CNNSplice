# -*- coding: utf-8 -*-
"""
Created on Tue Nov 9 17:24:14 2021
This should be executed on google coolab as the dependencies are 
no longer in support in other environment.
Install the dependencies on Google coolab
%tensorflow_version 1.x  -- this is to backdate the tensorflow version
!pip uninstall -y h5py
!pip install h5py==2.10.0
!pip install deeplift
!pip install simdna==0.4.3.2


@author: victorakpokiro
"""

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
import numpy as np
np.set_printoptions(threshold=np.inf)
from collections import OrderedDict
from keras.models import load_model
import pandas as pd

encoded_label1 = np.loadtxt('label3_1_encoded.txt')
encoded_label1 = np.loadtxt('all_acceptor_at_lbl')
encoded_label1 = encoded_label1.reshape(-1, 400, 5)



def read_seq(filepos, fileneg):
	fpos = open(filepos, "r")
	fneg = open(fileneg, "r")
	fpos_lines = fpos.readlines()
	fneg_line = fneg.readlines()
	seq_datapos = [seq.rstrip("\n")[:400] for seq in fpos_lines[:50]]
	seq_dataneg = [seq.rstrip("\n")[:400] for seq in fneg_line[:50]]
	seq_data = seq_datapos + seq_dataneg
	fpos.close()
	fneg.close()
	return seq_data


list_name = ["hs", "at", "oriza", "d_mel", "c_elegans"]

for datatype in list_name:
	seq_data = read_seq(f"./positive_DNA_seqs_acceptor_{datatype}.fa", f"./negative_DNA_seqs_acceptor_{datatype}.fa")
	f = open(f"./new_file/acceptor/acceptor_{datatype}.fa", "w")
	for i in seq_data:
		f.write(str(i))
		f.write("\n")
	f.close()


for datatype in list_name:
	seq_data = read_seq(f"./positive_DNA_seqs_donor_{datatype}.fa", f"./negative_DNA_seqs_donor_{datatype}.fa")
	f = open(f"./new_file/donor/donor_{datatype}.fa", "w")
	for i in seq_data:
		f.write(str(i))
		f.write("\n")
	f.close()


def data_seq(file):
	with open(file, 'r') as fl:
		all_lines = fl.readlines()
		seq_data = [seq.rstrip("\n")[:400] for seq in all_lines[:100]]
		# seq_datapos = [seq.rstrip("\n")[:400] for seq in all_linespos[:50]]
		# seq_dataneg = [seq.rstrip("\n")[:400] for seq in all_linesneg[:50]]
		#seq_data = seq_datapos + seq_dataneg
	return seq_data



def one_hot_encode_along_channel_axis(sequence):
	to_return = np.zeros((len(sequence),4), dtype=np.int8)
	seq_to_one_hot_fill_in_array(zeros_array=to_return,
								 sequence=sequence, one_hot_axis=1)

	#print(sequence)
	return to_return


def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
	assert one_hot_axis==0 or one_hot_axis==1
	if (one_hot_axis==0):
		assert zeros_array.shape[1] == len(sequence)
	elif (one_hot_axis==1): 
		assert zeros_array.shape[0] == len(sequence)
	#will mutate zeros_array
	for (i,char) in enumerate(sequence):
		if (char=="A" or char=="a"):
			char_idx = 0
		elif (char=="C" or char=="c"):
			char_idx = 1
		elif (char=="G" or char=="g"):
			char_idx = 2
		elif (char=="T" or char=="t"):
			char_idx = 3
		elif (char=="N" or char=="n"):
			continue #leave that pos as all 0's
		else:
			raise RuntimeError("Unsupported character: "+str(char))
		if (one_hot_axis==0):
			zeros_array[char_idx,i] = 1
		elif (one_hot_axis==1):
			zeros_array[i,char_idx] = 1

filepos = 'positive_DNA_seqs_acceptor_at.fa'
fileneg = 'negative_DNA_seqs_acceptor_at.fa'
# sequences = read_seq('positive_DNA_seqs_acceptor_c_elegans.fa')
sequences = read_seq(filepos, fileneg)
onehot_data = np.array([one_hot_encode_along_channel_axis(seq) for seq in sequences])

for j in range(8,9):
	print('======================model:CNN_1D_exclude_transcript_%f======================'%j)

	deeplift_model =kc.convert_model_from_saved_files('CNN_at_'+str(j)+'.h5', nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault)

	find_scores_layer_idx = 0
	deeplift_contribs_func = deeplift_model.get_target_contribs_func(find_scores_layer_idx=find_scores_layer_idx, target_layer_idx=-2)

	background = OrderedDict([('A', 0.3), ('C', 0.2), ('G', 0.2), ('T', 0.3), ('N', 0)])
	scores = np.array(deeplift_contribs_func(task_idx=2,
										  input_data_list=[encoded_label1],
										  input_references_list=[
										  np.array([background['A'],
													background['C'],
													background['G'],
													background['T'],
													background['N']])[None,None,:]],
										  batch_size=10,
										  progress_update=1000))


	scores = scores[:,:,:4]
	print("length",len(scores))
	final_score = scores[0,100:300]
	for i in range(1,100):
		final_score = final_score + scores[i,100:300]

	final_score = np.around(final_score, decimals=3)
	print(len(final_score))



	idx = 0
	scores_for_idx = scores[idx]
	original_onehot = onehot_data[idx]
	print(scores_for_idx.shape)
	print(scores_for_idx[:,None].shape)
	scores_for_idx = original_onehot*scores_for_idx[:,None]

	print(scores_for_idx.shape)
	print(len(scores[0]))
	arr = scores[0][150:250].transpose()
	df = pd.DataFrame(arr)
	df.to_csv("acceptor_at.csv")
	# scores[0][150:250].transpose().tofile('data2.csv')
	#np.savetxt('testscore.txt', scores[0])
	# np.savetxt('testscorepostt.txt', scores[0][150:250].transpose())
	#print(scores[0][150:250].transpose())

	from deeplift.visualization import viz_sequence

	viz_sequence.plot_weights(scores[0][150:250], subticks_frequency=10)  
 