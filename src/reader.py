# Here we will have the functions that will read in the data and also manipulate it

import pandas as pd
import numpy as np


def reader(filename):
    data = pd.read_csv("../data/"+filename)
    return data


def student_splitter(dataframe):
    ID = {}
    for id in dataframe.SubjectID.unique():
        ID[id] = dataframe.loc[dataframe['SubjectID'] == id]
    return ID


def video_splitter(df, split_id):
    return df.loc[df["VideoID"] != split_id], df.loc[df["VideoID"] == split_id]


def student_remove(dataframe, value):
    return dataframe[dataframe.SubjectID != value]


def fft_on_pandas(dataframe, columns):
    for column in columns:
        col = dataframe[column].tolist()
        dataframe[column] = np.abs(np.fft.fft(col)) ** 2
    return dataframe
