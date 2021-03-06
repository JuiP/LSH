import numpy as np
import pandas as pd 
import pickle
from collections import defaultdict
import copy 
import sys
from operator import itemgetter
from random import randint 
import time

def load(doc) :
    '''
    Load the pickle file
    '''
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
    shingles_dict=defaultdict(lambda : set([]))
    for doc in range (0,num_of_doc) :
        for x in range (0,len(data[doc])-k+1) :
            shingles_dict[data[doc][x:x+k]].add(doc)
    print("Shingling done")
    return shingles_dict

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
    print("initialization of Signature matrix done")
    # Has keys as the hash function and values as list for all documents

    for row in range (0, len(shingles_list)): # Keys
        hashes = genhash(len(shingles_list), num, row, func)
        for col in shingles[shingles_list[row]] :
            for n in range (0, num):
                if hashes[n] < signature_mat[n][col]:
                    signature_mat[n][col] = hashes[n]
    print("Signature Matrix created")

    signature_mat_list = []
    for key,value in signature_mat.items():
        signature_mat_list+=[value]
    signature_mat_list = np.array(signature_mat_list)
    return signature_mat_list
    

def L2_norm(x,y):
        '''This function is used to normalize a vector length using L2 norm '''
        return sum(pow((x[i] - y[i]), 2) for i in range(len(x))) ** (1/2)
    
def cosine_similarity(x,y):
    '''
    Computes the cosine similarity between two vectors
    '''
    numerator=0
    zeroes=np.zeros(len(x))
    for i in range(len(x)):
        numerator=numerator+(x[i]*y[i])
    A = L2_norm(x,zeroes)
    B = L2_norm(y,zeroes)
    return numerator/ (A*B)

def LSH(signature_mat, b, rows,num_docs):
    '''It is responsible for the local sensitive hashing. It divides the signature matrix into bands
       and documents having the same hashed value in a certain band are put into same bucket
       This function takes parameters:
       signature_mat : The Signature matrix obtained after minhashing
       b: number of bands in which signature matrix is divided
       rows: number of rows each band has
       num_docs: the number of documents in the corpus
       It returns two values:
       buckets: An array of dictionaries which holds the hashed vectors for each band
       hashed:It is the mapping using which docid was hashed into buckets
    '''	
    buckets=np.full(b,{})
    hashed=np.zeros((num_docs,b),dtype=int)
    for  i in range(b):
        for j in range(num_docs):
            l=signature_mat[int(i*rows):int((i+1)*rows), j]
            h=hash(tuple(l))
            if buckets[i].get(h):
                buckets[i][h].append(j)
            else:
                buckets[i][h]=[j]
            hashed[j][i]=h
    return hashed,buckets

    
def query_processing(hashed, buckets,signature_mat,query,t):
    '''This function is used to find the similar documents for a query within the same bucket
       obtained from LSH.
       The metric for search is Cosine Similarity
       The various parameters are
       hashed:It is the mapping using which docid was hashed into buckets
       buckets: An array of dictionaries which holds the hashed vectors for each band
       signature_mat: The Signature matrix obtained after minhashing
       query: the query document number to be searched in the corpus
       t: the threhold value for diciding similarity
       This function returns a sorted list of documents on the basis of similarity with the query document
    '''   
    c=[]
    for b,h in enumerate(hashed[query]):
        c.extend(buckets[b][h])
    c=set(c)
    sim_list=[]
    for doc in c:
        if doc==query:
            continue
        A = signature_mat[:,doc]
        B = signature_mat[:,query]
        sim = cosine_similarity(A,B)
        if(sim>=t):
            sim_list.append((round(sim, 3),doc))
    sim_list.sort(reverse=True)
    return sim_list

def main():
    data = load('human_data.obj')
    k = 5
    num_docs_initially=len(data)
    text=input("Enter sequence to be searched ")
    data[num_docs_initially]=text

    print("Time required for Shingling ")
    start_time = time.time()
    shingles = shingling(data , k)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Time required for Hashing ")
    start_time = time.time()
    number_of_hash_functions=100
    func = hashfunc(number_of_hash_functions, len(data))
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Time required for Signature Matrix ")
    start_time = time.time()
    signature_mat = signature_matrix(shingles, number_of_hash_functions , len(data), func)
    print("--- %s seconds ---" % (time.time() - start_time))

    b=5
    rows=int(number_of_hash_functions/b)
    threshold=0.9
    
    start_time = time.time()
    hashed, buckets=LSH(signature_mat,b,rows,len(data))
    print("Banding Done")
    val = len(data)-1
    print("Time required for LSH ")
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    print("Time required for query time ")
    sim_list=query_processing(hashed, buckets,signature_mat,val,threshold)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Similar DNA Patterns")
    for item in sim_list:
        print("Pattern number " + str(item[1]) + " with cosine similarity of " +str(item[0]) ) 
        print(data[item[1]])
    

#uncomment to run
main()
