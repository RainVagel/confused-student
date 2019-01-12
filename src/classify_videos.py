from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# import numpy as np
import lightgbm as lgb
from reader import *
from reproduce import *
import gc


class FeatureSelector():
    def __init__(self, data, labels=None):
        
        # Dataset and optional training labels
        self.data = data
        self.labels = labels

        if labels is None:
            print('No labels provided. Feature importance based methods are not available.')
        
        self.base_features = data.columns
        self.one_hot_features = None
        
        # Dataframes recording information about features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None
        
        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None
        
        # Dictionary to hold removal operations
        self.ops = {}
        
        self.one_hot_correlated = False

    def identify_collinear(self, correlation_threshold, one_hot=False):
        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot
        
         # Calculate the correlations between every column
        if one_hot:
            
            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)
            
            corr_matrix = pd.get_dummies(features).corr()

        else:
            corr_matrix = self.data.corr()
        
        self.corr_matrix = corr_matrix
    
        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        
        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:

            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]    

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                             'corr_feature': corr_features,
                                             'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index = True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop
        
        print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(self.ops['collinear']), self.correlation_threshold))
    
    def identify_zero_importance(self, task, eval_metric=None, n_iterations=10, early_stopping = True):
        if early_stopping and eval_metric is None:
            raise ValueError("""eval metric must be provided with early stopping. Examples include "auc" for classification or
                             "l2" for regression.""")
            
        if self.labels is None:
            raise ValueError("No training labels provided.")
        
        # One hot encoding
        features = pd.get_dummies(self.data)
        self.one_hot_features = [column for column in features.columns if column not in self.base_features]

        # Add one hot encoded data to original data
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        features = np.array(features)
        labels = np.array(self.labels).reshape((-1, ))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))
        
        print('Training Gradient Boosting Model\n')
        
        # Iterate through each fold
        for _ in range(n_iterations):

            if task == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1, objective = 'multiclass')

            elif task == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)

            else:
                raise ValueError('Task must be either "classification" or "regression"')
                
            # If training using early stopping need a validation set
            if early_stopping:
                
                train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size = 0.15)

                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric = 'logloss',
                          eval_set = [(valid_features, valid_labels)],
                          early_stopping_rounds = 100, verbose = -1)
                
                # Clean up memory
                gc.enable()
                del train_features, train_labels, valid_features, valid_labels
                gc.collect()
                
            else:
                model.fit(features, labels)

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
        
        to_drop = list(record_zero_importance['feature'])

        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop
        
        print('\n%d features with zero importance after one-hot encoding.\n' % len(self.ops['zero_importance']))

    def identify_low_importance(self, cumulative_importance):
        self.cumulative_importance = cumulative_importance
        
        # The feature importances need to be calculated before running
        if self.feature_importances is None:
            raise NotImplementedError("""Feature importances have not yet been determined. 
                                         Call the `identify_zero_importance` method first.""")
            
        # Make sure most important features are on top
        self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = self.feature_importances[self.feature_importances['cumulative_importance'] > cumulative_importance]

        to_drop = list(record_low_importance['feature'])

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop
    
        print('%d features required for cumulative importance of %0.2f after one hot encoding.' % (len(self.feature_importances) -
                                                                            len(self.record_low_importance), self.cumulative_importance))
        print('%d features do not contribute to cumulative importance of %0.2f.\n' % (len(self.ops['low_importance']),
                                                                                               self.cumulative_importance))
        
    def identify_missing(self, missing_threshold):
        """Find the features with a fraction of missing values above `missing_threshold`"""
        
        self.missing_threshold = missing_threshold

        # Calculate the fraction of missing in each column 
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns = {'index': 'feature', 0: 'missing_fraction'})

        # Sort with highest number of missing values on top
        self.missing_stats = self.missing_stats.sort_values('missing_fraction', ascending = False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns = 
                                                                                                               {'index': 'feature', 
                                                                                                                0: 'missing_fraction'})

        to_drop = list(record_missing['feature'])

        self.record_missing = record_missing
        self.ops['missing'] = to_drop
        
        print('%d features with greater than %0.2f missing values.\n' % (len(self.ops['missing']), self.missing_threshold))
        

def student_independent(data, target):
    clf = GaussianNB()
    scores = {}
    for key in data.SubjectID.unique():
        training, testing = student_independent_splitter(data, key)

        training = rework_data(training)
        testing = rework_data(testing)

        training_X, training_Y, testing_X, testing_Y = set_splitter(training, testing, target)

        # Normalization, norm='l1' or 'l2', normalization gives better results
        # than standardization
        normalizer = preprocessing.Normalizer(norm='l1').fit(training_X)
        training_X = normalizer.transform(training_X)
        testing_X = normalizer.transform(testing_X)

        clf = clf.partial_fit(training_X, training_Y, [0,1,2,3,4,5,6,7,8,9])
        predicted = clf.predict(testing_X)
        score = accuracy_score(testing_Y, predicted)
        scores[key] = score
    return scores


# Tried to split each video into windows of different sizes and take average of these windows but accuracy got worse
# Switching to mean value of each column of each video of each student
def rework_data(data):
    '''
    new = pd.DataFrame(columns=data.columns)

    for subject in data.SubjectID.unique():
        subject_data = data.loc[data["SubjectID"] == subject]

        for video in subject_data.VideoID.unique():
            df_temp = subject_data.loc[subject_data["VideoID"] == video]
            split_datasets = np.array_split(df_temp, 2)

            for df in split_datasets:
                row = {}

                for column in df.columns:
                    row[column] = df[column].mean()

                new = new.append(row, ignore_index=True)
    '''
    new = pd.DataFrame(columns=data.columns)
    for subject in data.SubjectID.unique():
        subject_data = data.loc[data["SubjectID"] == subject]
        for video in subject_data.VideoID.unique():
            df_temp = subject_data.loc[subject_data["VideoID"] == video]
            row = {}
            for column in df_temp.columns:
                row[column] = df_temp[column].mean()

            new = new.append(row, ignore_index=True)
    return new


# read in the data
df = reader("EEG_data.csv")


# student id 2 data might be corrupt.
df = student_remove(df, 2)


# make VideoID column the last column (for one hot encoding purposes)
column_to_relocate = df['VideoID']
df = df.drop(columns=['VideoID'])
df['VideoID'] = column_to_relocate


'''
# renaming the columns to 4-digit integer values
# necessary to avoid an error with one hot encoding
# 'SubjectID' = 1000
# 'Attention' = 1001
# ... 
# 'VideoID' = 1014
for col in df.columns:
    for new_col in range(1000, 1015):
        if new_col not in df.columns:
            df = df.rename({col: new_col}, axis='columns')


# one hot encoding
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df)
df = enc.transform(df)
'''



# Using the feature_selector module to remove low_importance features.
# Discovered that either predefinedlabel or user-definedlabeln can be removed at the 
# same time without losing accuracy but not both.
fs = FeatureSelector(data=df[['Attention','Mediation','Raw','Delta','Theta','Alpha1','Alpha2','Beta1',
                              'Beta2','Gamma1','Gamma2','predefinedlabel','user-definedlabeln']], labels=df['VideoID'])
fs.identify_collinear(correlation_threshold=0.95)
fs.identify_zero_importance(task='classification',
                                eval_metric='auc',
                                n_iterations=10,
                                early_stopping=True)
fs.identify_low_importance(cumulative_importance=0.99)
fs.identify_missing(missing_threshold=0.6)
df_predefined = df.drop(columns=fs.ops["low_importance"])
# user-definedlabeln is removed

independent_scores = student_independent(df_predefined, "VideoID")
print(independent_scores)
independent_scores_avg = sum(independent_scores.values()) / float(len(independent_scores.values()))
print(independent_scores_avg)