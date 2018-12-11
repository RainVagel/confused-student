# Here we will have the functions that will read in the data and also manipulate it

import pandas as pd


def reader(filename):
    data = pd.read_csv("../data/"+filename)
    return data


def student_splitter(dataframe):
    ID = {}
    for id in dataframe.SubjectID.unique():
        ID[id] = dataframe.loc[dataframe['SubjectID'] == id]
    return ID
