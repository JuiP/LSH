import numpy as np
import pandas as pd 
import pickle
from collections import defaultdict

def load(doc) :
    file = open(doc,'rb')
    data = pickle.load(file)
    file.close()
    return data

def shingling(data , k) :
    num_of_doc = len(data)
    listofzeros = [0] * num_of_doc
    shingles_dict=defaultdict(lambda : listofzeros)
    for doc in range (0,num_of_doc) :
        for x in range (0,len(data[doc])-k) :
            shingles_dict[data[doc][x:x+k]][doc]=1
    
    filehandler = open("shingles.obj","wb")
    pickle.dump(dict(shingles_dict),filehandler)
    filehandler.close()

def main():
    data = load('human_data.obj')
    k = 1000
    shingling(data , k)

#uncomment to run
# main()