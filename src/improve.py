from src.reader import *
from feature_selector import feature_selector
from src.reproduce import mean_rows, set_splitter
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# import CSP
import tensorflow as tf
from tensorflow import keras


def student_dependent(data, target):
    students = student_splitter(data)
    scores = {}
    # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
    #                          n_estimators=500)
    clf = SVC(kernel="poly")
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

            # NN model
            # model = keras.Sequential()
            # model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(10,)))
            # model.add(keras.layers.Dense(50, activation=tf.nn.relu))
            # model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
            #
            # model.compile(optimizer=tf.train.AdamOptimizer(),
            #               loss="binary_crossentropy",
            #               metrics=["accuracy"])
            # history = model.fit(training_X,
            #                     training_Y,
            #                     epochs=50,
            #                     verbose=1)
            # results = model.evaluate(testing_X, testing_Y)
            # print(results)
            # student_scores.append(results[1])

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

    # dependent_scores = student_dependent(df, "predefinedlabel")
    # print(dependent_scores)
    # dependent_scores_avg = sum(dependent_scores.values()) / float(len(dependent_scores.values()))
    # print(dependent_scores_avg)


if __name__ == '__main__':
    main()
