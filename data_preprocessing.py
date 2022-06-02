
import numpy as np

# def encodes(seq): 

# 	'''

# 	ONE-HOT ENCODING

# 	'''

# 	count = 1

# 	one_hot = {
# 		'A':np.array([1,0,0,0]).reshape(1,-1),
# 		'C':np.array([0,1,0,0]).reshape(1,-1),
# 		'G':np.array([0,0,1,0]).reshape(1,-1),
# 		'T':np.array([0,0,0,1]).reshape(1,-1),
# 		'_':np.array([0,0,0,0]).reshape(1,-1), # unknown nucleotides
# 	}


# 	while (count <= 400):
# 		encoded_seq = []
# 		for nucleo in list(seq):
# 			if nucleo.upper() in 'ACGT':
# 				encoded_seq.append(one_hot[nucleo.upper()])
# 			else:
# 				encoded_seq.append(one_hot['_'])
# 			count += 1
# 		return encoded_seq




def encodes(seq): 
	'''

	ONE-HOT ENCODING

	'''
	encoded_seq = np.zeros(1600,int)
	for j in range(400):
		if seq[j] == 'A' or seq[j] == 'a':
			encoded_seq[j*4] = 1
			encoded_seq[j*4+1] = 0
			encoded_seq[j*4+2] = 0
			encoded_seq[j*4+3] = 0

		elif seq[j] == 'C' or seq[j] == 'c':
			encoded_seq[j*4] = 0
			encoded_seq[j*4+1] = 1
			encoded_seq[j*4+2] = 0
			encoded_seq[j*4+3] = 0

		elif seq[j] == 'G' or seq[j] == 'g':
			encoded_seq[j*4] = 0
			encoded_seq[j*4+1] = 0
			encoded_seq[j*4+2] = 1
			encoded_seq[j*4+3] = 0

		elif seq[j] == 'T' or seq[j] == 't':
			encoded_seq[j*4] = 0
			encoded_seq[j*4+1] = 0
			encoded_seq[j*4+2] = 0
			encoded_seq[j*4+3] = 1

		else:
			encoded_seq[j*4] = 0
			encoded_seq[j*4+1] = 0
			encoded_seq[j*4+2] = 0
			encoded_seq[j*4+3] = 0

	return encoded_seq



def write2file(out_file_loc, seq_data_lst):

	with open(out_file_loc, 'w') as f1le:

		for seq in seq_data_lst:
			for value in seq:
				f1le.write(str(value))
				f1le.write(" ")
			f1le.write('\n')

	return

# def writelbl(out_file_loc, range_num):
# 	with open(out_file_loc, 'w') as file:
# 		for i in range(range_num):
# 			file.write(str(1))
# 			file.write(" ")
		# file.write('\n')


def writelbl(out_file_loc, seq_lbl_lst):
	with open(out_file_loc, 'w') as file:
		for lbl in seq_lbl_lst:
			file.write(str(lbl))
			file.write(" ")



def read_5000(input_fl, output_lst):
	with open(input_fl, "r") as file:
		
		for i in range(5000):
			dataline = file.readline().strip('\n')
			encode_seq = encodes(dataline)
			output_lst.append(encode_seq)


def read_7500(input_fl, output_lst):
	with open(input_fl, "r") as file:
		
		for i in range(7500):
			dataline = file.readline().strip('\n')
			encode_seq = encodes(dataline)
			output_lst.append(encode_seq)


def read_2500(input_fl, output_lst):
	with open(input_fl, "r") as file:
		
		for i in range(2500):
			dataline = file.readline().strip('\n')
			encode_seq = encodes(dataline)
			output_lst.append(encode_seq)


def read_15000(input_fl, output_lst):
	with open(input_fl, "r") as file:
		
		for i in range(15000):
			dataline = file.readline().strip('\n')
			encode_seq = encodes(dataline)
			output_lst.append(encode_seq)


def read_22500(input_fl, output_lst):
	with open(input_fl, "r") as file:
		
		for i in range(22500):
			dataline = file.readline().strip('\n')
			encode_seq = encodes(dataline)
			output_lst.append(encode_seq)


def read_7500(input_fl, output_lst):
	with open(input_fl, "r") as file:
		
		for i in range(7500):
			dataline = file.readline().strip('\n')
			encode_seq = encodes(dataline)
			output_lst.append(encode_seq)




def process_seq_file(inp_file_loc_pos, inp_file_loc_neg, out_file_train, out_file_test, out_lbl_train, out_lbl_test, out_file_all, out_lbl_all):
# def process_seq_file(inp_file_loc_pos, inp_file_loc_neg, out_file, out_lbl):
	count = 1
	all_sequence = []
	train_sequence = []
	test_sequence = []
	sequences = []
	label = []

	#balanced 
	# read_5000(inp_file_loc_pos, all_sequence)
	# read_5000(inp_file_loc_neg, all_sequence)

	#inbalanced
	# read_7500(inp_file_loc_pos, all_sequence)
	# read_2500(inp_file_loc_neg, all_sequence)


	#balanced 
	# read_15000(inp_file_loc_pos, all_sequence)
	# read_15000(inp_file_loc_neg, all_sequence)

	#inbalanced
	read_22500(inp_file_loc_pos, all_sequence)
	read_7500(inp_file_loc_neg, all_sequence)

	#balanced ---- train test sequence data
	# train_sequence = all_sequence[:3500] + all_sequence[5000:8500]
	# test_sequence = all_sequence[3500:5000] + all_sequence[8500:10000]

	#inbalanced ---- train test sequence data
	# train_sequence = all_sequence[:5625] + all_sequence[7500:9375]
	# test_sequence = all_sequence[5625:7500] + all_sequence[9375:10000]


	#balanced ---- train test sequence data
	# train_sequence = all_sequence[:10500] + all_sequence[15000:25500]
	# test_sequence = all_sequence[10500:15000] + all_sequence[25500:30000]

	#inbalanced ---- train test sequence data
	train_sequence = all_sequence[:16875] + all_sequence[22500:28125]
	test_sequence = all_sequence[16875:22500] + all_sequence[28125:30000]

	sequences = train_sequence + test_sequence



	#balanced train test sequence label
	# all_seq_label = [0]*5000 + [1]*5000

	#inbalanced train test sequence label
	# all_seq_label = [0]*7500 + [1]*2500

	#balanced train test sequence label
	# all_seq_label = [0]*15000 + [1]*15000

	#inbalanced train test sequence label
	all_seq_label = [0]*22500 + [1]*7500



	#balaned
	# train_seq_lbl = all_seq_label[:3500] + all_seq_label[5000:8500]
	# test_seq_lbl = all_seq_label[3500:5000] + all_seq_label[8500:10000]

	#inbalanced
	# train_seq_lbl = all_seq_label[:5625] + all_seq_label[7500:9375]
	# test_seq_lbl = all_seq_label[5625:7500] + all_seq_label[9375:10000]

	#balanced
	# train_seq_lbl = all_seq_label[:10500] + all_seq_label[15000:25500]
	# test_seq_lbl = all_seq_label[10500:15000] + all_seq_label[25500:30000]

	#inbalanced
	train_seq_lbl = all_seq_label[:16875] + all_seq_label[22500:28125]
	test_seq_lbl = all_seq_label[16875:22500] + all_seq_label[28125:30000]


	label = train_seq_lbl + test_seq_lbl

		
	#balanced --- write to file --train test sequence data
	# write2file(out_file_train, train_sequence)
	# write2file(out_file_test, test_sequence)


	#inbalanced --- write to file --train test sequence data
	write2file(out_file_train, train_sequence)
	write2file(out_file_test, test_sequence)

	
	write2file(out_file_all, all_sequence)


	#balanced --- write to file --train test sequence label
	# writelbl(out_lbl_train, train_seq_lbl)
	# writelbl(out_lbl_test, test_seq_lbl)


	#inbalanced --- write to file --train test sequence label
	writelbl(out_lbl_train, train_seq_lbl)
	writelbl(out_lbl_test, test_seq_lbl)

	writelbl(out_lbl_all, all_seq_label)



datatype = "imbalanced"

def main():

	seq = "acceptor"
	list_name = ["hs", "c_elegans", "d_mel", "at", "oriza"]

	for filename in list_name:
		inp_file_loc_pos = f"./Data/positive_DNA_seqs_{seq}_{filename}.fa"
		inp_file_loc_neg = f"./Data/negative_DNA_seqs_{seq}_{filename}.fa"
		output_train = f"./review/data/{datatype}/{filename}/train_{seq}_{filename}"
		output_test = f"./review/data/{datatype}/{filename}/test_{seq}_{filename}"
		out_file_all = f"./review/data/{datatype}/{filename}/all_{seq}_{filename}"
		out_lbl_all = f"./review/data/{datatype}/{filename}/all_{seq}_{filename}_lbl"
		train_lbl = f"./review/data/{datatype}/{filename}/train_{seq}_{filename}_lbl"
		test_lbl = f"./review/data/{datatype}/{filename}/test_{seq}_{filename}_lbl"
		process_seq_file(inp_file_loc_pos, inp_file_loc_neg, output_train, output_test, train_lbl, test_lbl, out_file_all, out_lbl_all)
		# process_seq_file(inp_file_loc_pos, inp_file_loc_neg, out_file, out_lbl)


	seq = "donor"

	for filename in list_name:
		inp_file_loc_pos = f"./Data/positive_DNA_seqs_{seq}_{filename}.fa"
		inp_file_loc_neg = f"./Data/negative_DNA_seqs_{seq}_{filename}.fa"
		output_train = f"./review/data/{datatype}/{filename}/train_{seq}_{filename}"
		output_test = f"./review/data/{datatype}/{filename}/test_{seq}_{filename}"
		out_file_all = f"./review/data/{datatype}/{filename}/all_{seq}_{filename}"
		out_lbl_all = f"./review/data/{datatype}/{filename}/all_{seq}_{filename}_lbl"
		train_lbl = f"./review/data/{datatype}/{filename}/train_{seq}_{filename}_lbl"
		test_lbl = f"./review/data/{datatype}/{filename}/test_{seq}_{filename}_lbl"
		process_seq_file(inp_file_loc_pos, inp_file_loc_neg, output_train, output_test, train_lbl, test_lbl, out_file_all, out_lbl_all)
		# process_seq_file(inp_file_loc_pos, inp_file_loc_neg, out_file, out_lbl)


if __name__ == '__main__':
	main()