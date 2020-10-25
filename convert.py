import pandas as pd 
import re
import pickle

def converter() :  
    # reading given csv file and convert to a dictionary
    data = pd.read_csv("human_data.txt",header = None) 
    data_dict = data.to_dict()
    data_dict = data_dict[0]

    #removing unwanted characters from sequence
    for doc, seq  in data_dict.items():
        data_dict[doc]=  re.sub('[^a-zA-Z]+', '', seq)

    # store dataframe into pickle file 
    filehandler = open("human_data.obj","wb")
    pickle.dump(data_dict,filehandler)
    filehandler.close()

#uncomment to run
# converter()