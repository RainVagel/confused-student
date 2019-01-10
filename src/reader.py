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


def student_independent_splitter(df, split_id):
    return df.loc[df["SubjectID"] != split_id], df.loc[df["SubjectID"] == split_id]


def video_splitter(df, split_id):
    return df.loc[df["VideoID"] != split_id], df.loc[df["VideoID"] == split_id]


def student_remove(dataframe, value):
    return dataframe[dataframe.SubjectID != value]


def fft_on_pandas(dataframe, columns):
    for column in columns:
        col = dataframe[column].tolist()
        dataframe[column] = np.abs(np.fft.fft(col)) ** 2
    return dataframe


def reshape_samples(dataframe):
    min_samples = np.inf

    # First get the maximum number of samples that each subject with each video has
    for id in dataframe.SubjectID.unique():
        for video in dataframe.VideoID.unique():
            min_samples = min(min_samples,
                              len(dataframe.loc[(dataframe["SubjectID"] == id) & (dataframe["VideoID"] == video)]))

    # Create a new dataframe where each subject would have the same amount of samples for each video
    new_df = pd.DataFrame()
    for id in dataframe.SubjectID.unique():
        for video in dataframe.VideoID.unique():
            new_df = new_df.append(
                dataframe.loc[(dataframe["SubjectID"] == id) & (dataframe["VideoID"] == video)].iloc[:112],
                ignore_index=True)
    return new_df
