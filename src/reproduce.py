from sklearn.naive_bayes import GaussianNB
from src.reader import *
from sklearn.metrics import accuracy_score
import pandas as pd


def set_splitter(train, test, y_column):
    training_X_columns = [x for x in train.columns if x not in ["predefinedlabel", "user-definedlabeln",
                                                                   "SubjectID", "VideoID"]]
    training_X = train[training_X_columns]
    training_Y = train[y_column]
    testing_X_columns = [x for x in test.columns if x not in ["predefinedlabel", "user-definedlabeln",
                                                                  "SubjectID", "VideoID"]]
    testing_X = test[testing_X_columns]
    testing_Y = test[y_column]
    return training_X, training_Y, testing_X, testing_Y


def mean_rows(data):
    new = pd.DataFrame(columns=data.columns)
    for video in data.VideoID.unique():
        df_temp = data.loc[data["VideoID"] == video]
        row = {}
        for column in df_temp.columns:
            row[column] = df_temp[column].mean()
        new = new.append(row, ignore_index=True)
    return new


def compacter(data):
    confused = 0
    not_confused = 0
    for i in data:
        if i == 1:
            confused += 1
        else:
            not_confused += 1
    if confused > not_confused:
        return [1] * len(data)
    return [0] * len(data)


def subtract_mean(data):
    for column in ["Raw","Delta","Theta","Alpha1", "Alpha2","Beta1","Beta2","Gamma1","Gamma2"]:
        data[column] = data[column] - data[column].mean()
    return data


def student_dependent(data):
    students = student_splitter(data)
    scores = {}
    clf = GaussianNB()
    for key in students.keys():
        student_set = students[key]
        student_scores = []
        for split in student_set.VideoID.unique():
            training, testing = video_splitter(student_set, split)

            # Take the mean of all the rows so that each video will have 1 row.
            # Comment them out if you want to run on the non-aggregated data
            # training = mean_rows(training)
            # testing = mean_rows(testing)

            training_X, training_Y, testing_X, testing_Y = set_splitter(training, testing, "predefinedlabel")
            clf.fit(training_X, training_Y)
            predicted = clf.predict(testing_X)
            predicted = compacter(predicted)
            score = accuracy_score(testing_Y, predicted)
            student_scores.append(score)
        print(student_scores)
        scores[key] = sum(student_scores) / float(len(student_scores))
    return scores


def student_independent(data):
    pass


def main():
    df = reader("EEG_data.csv")
    df = subtract_mean(student_remove(df, 6))
    # df = fft_on_pandas(df, ["Attention","Mediation","Raw","Delta","Theta","Alpha1",
    #                         "Alpha2","Beta1","Beta2","Gamma1","Gamma2"])
    dependent_scores = student_dependent(df)
    print(dependent_scores)
    print(sum(dependent_scores.values()) / float(len(dependent_scores.values())))


if __name__ == '__main__':
    main()
