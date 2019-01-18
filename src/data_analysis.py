# -*- coding: utf-8 -*-
# Needs https://github.com/WillKoehrsen/feature-selector installed to be used
# To install, download it from Github, extract it and with "Anaconda Prompt"
# run "conda setup.py install"
#
# Requires:
# python==3.6+
# lightgbm==2.1.1
# matplotlib==2.1.2
# seaborn==0.8.1
# numpy==1.14.5
# pandas==0.23.1
# scikit-learn==0.19.1

from feature_selector import FeatureSelector
import src.reader as rdr
import pandas as pd
import matplotlib.pyplot as plt


pd.set_option('display.expand_frame_repr', False)

# Create training data and training labels for FeatureSelector
data = rdr.reader("EEG_data.csv")
training_label_predefined = data['predefinedlabel']
training_label_user_defined = data['user-definedlabeln']
training_labels = data[['predefinedlabel', 'user-definedlabeln']]
training_data = data.drop(columns = ['SubjectID', 'VideoID', 'predefinedlabel', 'user-definedlabeln'])

# Student independent 

# Define FeatureSelector
fs = FeatureSelector(data = training_data, labels = training_labels)
fs_predefined = FeatureSelector(data = training_data, labels = training_label_predefined)
fs_user_defined = FeatureSelector(data = training_data, labels = training_label_user_defined)

# 1. Find missing values
def find_missing_values():
    fs.identify_missing(missing_threshold=0.01)
    # "0 features with greater than 0.60 missing values."
    # "0 features with greater than 0.01 missing values."
    # No missing values at all.
    
#find_missing_values()


# 2. Single Unique Value
def find_single_unique_value():
    fs.identify_single_unique()
    # "0 features with a single unique value."
    # No single unique values.
    
#find_single_unique_value(fs)


# 3. Collinear (highly correlated) Features
def find_collinearity():
    fs.identify_collinear(correlation_threshold=0.95)
    fs.plot_collinear(plot_all=True)
    # "0 features with a correlation magnitude greater than 0.95."
    #  No highly correlated features. 
    ############# Save this plot! ###############
    
#find_collinearity(fs)


# 4. Zero Importance Features
# Need to seperate training labels to train
def find_zero_importance():
    fs_predefined.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                                n_iterations = 10, early_stopping = True)
    # "0 features with zero importance after one-hot encoding."
    
    fs_user_defined.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                                n_iterations = 10, early_stopping = True)
    # "0 features with zero importance after one-hot encoding."
    

find_zero_importance()
    
    
# 5. Plotting feature importances
def plot_feat_importances():
    fs_predefined.plot_feature_importances(threshold=0.95) 
    fs_user_defined.plot_feature_importances(threshold=0.95)
plt.figure(20)
plot_feat_importances()
    

# 6. Low importance features
def low_importance():
    fs_predefined.identify_low_importance(cumulative_importance = 1.0)
    fs_user_defined.identify_low_importance(cumulative_importance = 1.0)

#low_importance()
#low_importance_predefined = fs_predefined.ops['low_importance']
#low_importance_user_defined = fs_user_defined.ops['low_importance']
#print(low_importance_predefined)
#print(low_importance_user_defined)
#print(fs_predefined.feature_importances)
# There are actually no low importance features.

# Exploratory data analysis
# Student independent by predefined
    
data_analysis = data.drop(columns = ['SubjectID', 'VideoID'])
data_label_0 = data_analysis.loc[data_analysis.predefinedlabel == 0]
data_label_1 = data_analysis.loc[data_analysis.predefinedlabel == 1]

data_label_0_describe = data_label_0.describe()
data_label_1_describe = data_label_1.describe()


cdf = pd.concat([data_label_0['Theta'].reset_index(), data_label_1['Theta'].reset_index()], axis=1).drop(columns=['index'])
plt.figure(0)
cdf.boxplot()
plt.title("Student independent predefined: \nnot confused vs confused")
plt.show()
# Everything is basically the same!
# But this plot is pretty nice as a visual 


# Student independent by user defined
data_label_0_user = data_analysis.loc[data_analysis['user-definedlabeln'] == 0]
data_label_1_user = data_analysis.loc[data_analysis['user-definedlabeln'] == 1]

data_label_0_user_describe = data_label_0_user.describe()
data_label_1_user_describe = data_label_1_user.describe()


cdf = pd.concat([data_label_0_user['Theta'].reset_index(), data_label_1_user['Theta'].reset_index()], axis=1).drop(columns=['index'])
plt.figure(1)
cdf.boxplot()
plt.title("Student independent user-defined: \nnot confused vs confused")
plt.show()

# Comparison of predefined and userdefined
cdf = pd.concat([data_label_0['Theta'].reset_index(), data_label_0_user['Theta'].reset_index()], axis=1).drop(columns=['index'])
plt.figure(2)
cdf.boxplot()
plt.title("Student independent predefined vs user-defined")
plt.show()

# All same!


# Student dependent
students_describe = {}
students = rdr.student_splitter(data)
for key in students.keys():
    student_set = students[key].describe()
    students_describe[key] = student_set

# Predefined
student_0 = students[0]
student_0_label_0 = student_0.loc[student_0.predefinedlabel == 0]
student_0_label_1 = student_0.loc[student_0.predefinedlabel == 1]

cdf = pd.concat([student_0_label_0['Theta'].reset_index(), student_0_label_1['Theta'].reset_index()], axis=1).drop(columns=['index'])
plt.figure(3)
cdf.boxplot()
plt.title("Student 1 predefined:\nnot confused vs confused")
plt.show()

# Another student

student_1 = students[2]
student_1_label_0 = student_1.loc[student_1.predefinedlabel == 0]
student_1_label_1 = student_1.loc[student_1.predefinedlabel == 1]

cdf = pd.concat([student_1_label_0['Theta'].reset_index(), student_1_label_1['Theta'].reset_index()], axis=1).drop(columns=['index'])
plt.figure(4)
cdf.boxplot()
plt.title("Student 3 predefined:\nnot confused vs confused")
plt.show()

# Comparison of students
cdf = pd.concat([student_0_label_0['Theta'].reset_index(), student_1_label_0['Theta'].reset_index()], axis=1).drop(columns=['index'])
plt.figure(5)
cdf.boxplot()
plt.title("Student 1 vs 3: not confused")
plt.show()

# Not using main(), because of Spyder's variable explorer
# If needed, uncomment this part and comment the previous part of code and copy
# it here
#def main():
    

#if __name__ == '__main__':
#    main()
