import numpy as np
import pandas as pd 
import pickle
from collections import defaultdict
import copy 
import sys
from random import randint 

def load(doc) :
    file = open(doc,'rb')
    data = pickle.load(file)
    file.close()
    return data

def shingling(data , k) :
    '''
    shingles_dict stores the input matrix.
    Keys of the dictionary are the shingles and the value is matrix values (rows)
    '''
    num_of_doc = len(data)
    listofzeros = [0] * num_of_doc
    shingles_dict=defaultdict(lambda : copy.deepcopy(listofzeros))
    for doc in range (0,num_of_doc) :
        for x in range (0,len(data[doc])-k+1) :
            shingles_dict[data[doc][x:x+k]][doc]=1
    print("Shingling done")
    return shingles_dict
    # filehandler = open("shingles.obj","wb")
    # pickle.dump(dict(shingles_dict),filehandler)
    # filehandler.close()
    # return shingles_dict

def hashfunc(num, length):
	'''
	num is the number of hash functions to be created.
	Returns list of (a,b) pairs.
	Hash function is of the form (ax+b)modlength
	Here a and b is always smaller than the number of shingles.
	'''
	a = randint(1,length)
	b = randint(1,length)
	functions = {(a,b)}
	while len(functions) < num:
		a = randint(1,length)
		b = randint(1,length)
		functions.update({(a,b)})
	#print(list(functions))
	return list(functions)

def genhash(length, num, x, func):
	'''
	length is the number of shingles.
	num is the number of hash functions.
	x is the row for which hash function value is to be calculated.
	func is the list returned by hashfunc()
	Returns the list containing hash functions value for row x.
	'''
	hashes = []
	for i in range (0, num):
		val = (func[i][0] * x + func[i][1]) % length
		hashes.append(val)
	return hashes


def signature_matrix(shingles, num, no_of_doc, func):
	'''
	shingles is the Input matrix with value of dictionary as the shingles.
	num is the number of minhash functions to be generated.
	no_of_doc is the number of documents in data.
	func is the list returned by hashfunc()
	'''
	shingles_list = list(shingles.keys())
	listofinfinity = [sys.maxsize] * no_of_doc
	signature_mat = {}
	for x in range (0, num):
		signature_mat[x] = copy.deepcopy(listofinfinity)
	print("initialization of signature_mat done")
	# Has keys as the hash function and values as list for all documents

	# row refers to row of input matrix (conceptually)
	for row in range (0, len(shingles_list)):
		hashes = genhash(len(shingles_list), num, row, func)
		for col in range (0, no_of_doc):
			if shingles[shingles_list[row]][col] == 1:
				for n in range (0, num):
					if hashes[n] < signature_mat[n][col]:
						signature_mat[n][col] = hashes[n]
	print("Signature Matrix created")
	return signature_mat

def main():
    data = load('human_data.obj')
    k = 5
    shingles = shingling(data , k)
    func = hashfunc(100, len(data))
    signature_mat = signature_matrix(shingles, 100, len(data), func)
    
    # --------------Debugging--------------------------------
    # shingles = \
    # { "0": [0,1,1,0],
    #   "1": [0,0,1,1],
    #   "2": [1,0,0,0],
    #   "3": [0,1,0,1],
    #   "4": [0,0,0,1],
    #   "5": [1,1,0,0],
    #   "6": [0,0,1,0]
    # }
    # func = hashfunc(3, 4)
    # signature_mat = signature_matrix(shingles, 3, 4, func)
    # print(signature_mat)
    # --------------------------------------------------------

    # filehandler = open("signature_matrix.obj","wb")
    # pickle.dump(dict(signature_mat),filehandler)
    # filehandler.close()
    

#uncomment to run
main()