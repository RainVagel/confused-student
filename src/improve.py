from src.reader import *
from feature_selector import feature_selector
from src.reproduce import mean_rows, set_splitter
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from src.plotter import reproduce_plotter, formatter
import os

# import CSP
import tensorflow as tf
from tensorflow import keras


def student_dependent_capstone(data, target):
    students = student_splitter(data)
    scores = {}
    for key in students.keys():
        student_set = students[key]
        student_scores = []
        for split in student_set.VideoID.unique():
            scaler = MinMaxScaler()
            x = student_set.iloc[:, 2:-2]
            scaler.fit(x)
            x = scaler.transform(x)
            x_mean = x.mean()
            x_std = x.std()
            x = (x - x_mean)/x_std
            student_set.is_copy = False
            student_set.iloc[:, 2:-2] = x

            training, testing = video_splitter(student_set, split)
            training_X, training_Y, testing_X, testing_Y = set_splitter(training, testing, target)

            model = keras.Sequential()
            model.add(keras.layers.Dense(300, activation="relu", input_shape=(11,)))
            model.add(keras.layers.Dropout(0.4))
            model.add(keras.layers.Dense(300, activation="relu"))
            model.add(keras.layers.Dropout(0.4))
            model.add(keras.layers.Dense(300, activation="relu"))
            model.add(keras.layers.Dropout(0.4))
            model.add(keras.layers.Dense(300, activation="relu"))
            model.add(keras.layers.Dropout(0.4))
            model.add(keras.layers.Dense(1, activation="sigmoid"))

            model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
            hist = model.fit(training_X, training_Y, epochs=100, batch_size=32, validation_split=0.1)
            score = model.evaluate(testing_X, testing_Y, batch_size=32)
            print("Accuracy:", score[1])
            student_scores.append(score[1])
        scores[key] = sum(student_scores) / float(len(student_scores))
    return scores


def student_dependent(data, target):
    students = student_splitter(data)
    scores = {}
    # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
    #                          n_estimators=500)
    # clf = SVC(kernel="poly")
    clf = LogisticRegressionCV(cv=3)
    for key in students.keys():
        student_set = students[key]
        student_scores = []
        for split in student_set.VideoID.unique():
            training, testing = video_splitter(student_set, split)

            # Take the mean of all the rows so that each video will have 1 row.
            # Comment them out if you want to run on the non-aggregated data
            # Using mean at the moment gives better results
            training = mean_rows(training)
            testing = mean_rows(testing)

            training_X, training_Y, testing_X, testing_Y = set_splitter(training, testing, target)

            # Scaler
            scaler = preprocessing.StandardScaler().fit(training_X)
            training_X = scaler.transform(training_X)
            testing_X = scaler.transform(testing_X)

            # Normalization, norm='l1' or 'l2', normalization gives better results
            # than standardization
            # normalizer = preprocessing.Normalizer(norm='l2').fit(training_X)
            # training_X = normalizer.transform(training_X)
            # testing_X = normalizer.transform(testing_X)

            clf = clf.fit(training_X, training_Y)
            predicted = clf.predict(testing_X)
            score = accuracy_score(testing_Y, predicted)
            student_scores.append(score)
        scores[key] = sum(student_scores) / float(len(student_scores))
    return scores


def student_independent(data, target):
    scores = {}
    scaler = MinMaxScaler()
    x = data.iloc[:, 2:-2]
    scaler.fit(x)
    x = scaler.transform(x)
    x_mean = x.mean()
    x_std = x.std()
    x = (x - x_mean) / x_std
    data.is_copy = False
    data.iloc[:, 2:-2] = x
    for key in data.SubjectID.unique():
        training, testing = student_independent_splitter(data, key)

        training_X, training_Y, testing_X, testing_Y = set_splitter(training, testing, target)

        # Normalization, norm='l1' or 'l2', normalization gives better results
        # than standardization
        # normalizer = preprocessing.Normalizer(norm='l1').fit(training_X)
        # training_X = normalizer.transform(training_X)
        # testing_X = normalizer.transform(testing_X)

        model = keras.Sequential()
        model.add(keras.layers.Dense(300, activation="relu", input_shape=(11,)))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(300, activation="relu"))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(300, activation="relu"))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(300, activation="relu"))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        hist = model.fit(training_X, training_Y, epochs=100, batch_size=32, validation_split=0.1)
        score = model.evaluate(testing_X, testing_Y, batch_size=32)
        scores[key] = score[1]
    return scores


def unshuffled_nn(df, path):
    print("Starting Student dependent predefinedlabels")
    dependent_scores = student_dependent_capstone(df, "predefinedlabel")
    dependent_final = formatter(dependent_scores, "Predefined")

    print("Starting student dependent user-definedlabels")
    dependent_scores_user = student_dependent_capstone(df, "user-definedlabeln")
    dependent_user_final = formatter(dependent_scores_user, "User-defined")

    dependent_performances = [dependent_final, dependent_user_final]

    reproduce_plotter(path, performances=dependent_performances,
                      labels=["Average"] + [str(int(x) + 1) for x in dependent_scores.keys()],
                      x_label="Students", y_label="Accuracy", title="Student dependent NN")

    print("Starting student independent predefinedlabels")
    independent_scores = student_independent(df, "predefinedlabel")
    independent_final = formatter(independent_scores, "Predefined")

    print("Starting student independent user-definedlabels")
    independent_scores_user = student_independent(df, "user-definedlabeln")
    independent_user_final = formatter(independent_scores_user, "User-defined")

    independent_performances = [independent_final, independent_user_final]
    reproduce_plotter(path, performances=independent_performances,
                      labels=["Average"] + [str(int(x) + 1) for x in independent_scores.keys()],
                      x_label="Students", y_label="Accuracy", title="Student independent NN")


def shuffled_nn(df, path):
    # In this method we will take traditional testing measures as in just some percentage of shuffled dataset
    pass


def main():
    df = student_remove(reader("EEG_data.csv"), 2)

    # Make all of the different samples have same amount of rows
    df = reshape_samples(df)

    # Path for the graphs
    os.chdir("..")
    path = os.curdir + "/img/"

    unshuffled_nn(df, path)


if __name__ == '__main__':
    main()
