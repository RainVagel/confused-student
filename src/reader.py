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


def video_splitter(df, split_id):
    return df.loc[df["VideoID"] != split_id], df.loc[df["VideoID"] == split_id]

# print(video_splitter(reader("EEG_data.csv"), 3))
