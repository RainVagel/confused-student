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
from src.plotter import reproduce_plotter
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

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            hist = model.fit(training_X, training_Y, epochs=100, batch_size=32, validation_split=0.1)
            score = model.evaluate(testing_X, testing_Y)
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

            # Stuff for tensorflow
            # training_Y = np.asarray(list(training_Y))
            # testing_Y = np.asarray(list(testing_Y))

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


def main():
    df = student_remove(reader("EEG_data.csv"), 2)

    # Make all of the different samples have same amount of rows
    df = reshape_samples(df)
    dependent_scores = student_dependent_capstone(df, "predefinedlabel")
    print(dependent_scores)
    dependent_scores_avg = sum(dependent_scores.values()) / float(len(dependent_scores.values()))
    print(dependent_scores_avg)
    # fs = feature_selector.FeatureSelector(data=df[["Attention", "Mediation", "Raw", "Delta", "Theta", "Alpha1",
    #                                                "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2"]],
    #                                       labels=df["predefinedlabel"])
    # fs.identify_collinear(correlation_threshold=0.95)
    # fs.identify_zero_importance(task='classification',
    #                             eval_metric='auc',
    #                             n_iterations=10,
    #                             early_stopping=True)
    # fs.identify_low_importance(cumulative_importance=0.99)
    # fs.identify_missing(missing_threshold=0.6)
    # df_predefined = df.drop(columns=fs.ops["low_importance"])
    #
    # dependent_scores = student_dependent(df_predefined, "predefinedlabel")
    # print(dependent_scores)
    # dependent_scores_avg = sum(dependent_scores.values()) / float(len(dependent_scores.values()))
    # print(dependent_scores_avg)
    # dependent_s = [dependent_scores_avg] + list(dependent_scores.values())
    # dependent_final_scores = ["Original", dependent_s]
    # trial_l = [0.48, 0.98, 0.87, 0.76, 0.65, 0.54, 0.43, 0.32, 0.21, 0.11]
    # trial = ["Trial", trial_l]
    # performances = [dependent_final_scores, trial]
    # print(performances)
    # os.chdir("..")
    # path = os.curdir
    # reproduce_plotter(path, performances=performances,
    #                   labels=["Average"] + [str(int(x)+1) for x in dependent_scores.keys()],
    #                   x_label="Students", y_label="Accuracy", title="Katsetame")


if __name__ == '__main__':
    main()
