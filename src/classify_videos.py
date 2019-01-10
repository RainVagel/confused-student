from sklearn.naive_bayes import GaussianNB
from reader import *
from sklearn import preprocessing
from reproduce import *
from feature_selector import FeatureSelector
from reproduce import *

df = reader("EEG_data.csv")

# student id 2 data might be corrupt.
df = student_remove(df, 2)


# Using the feature_selector module to remove low_importance features.
fs = FeatureSelector(data=df[["Attention","Mediation","Raw","Delta","Theta","Alpha1",
                            "Alpha2","Beta1","Beta2","Gamma1","Gamma2",'predefinedlabel',
                            'user-definedlabeln']], labels=df["VideoID"])
fs.identify_collinear(correlation_threshold=0.95)
fs.identify_zero_importance(task='classification',
                                eval_metric='auc',
                                n_iterations=10,
                                early_stopping=True)
fs.identify_low_importance(cumulative_importance=0.99)
fs.identify_missing(missing_threshold=0.6)
df_predefined = df.drop(columns=fs.ops["low_importance"])

independent_scores = student_independent(df_predefined, "VideoID")
print(independent_scores)
independent_scores_avg = sum(independent_scores.values()) / float(len(independent_scores.values()))
print(independent_scores_avg)