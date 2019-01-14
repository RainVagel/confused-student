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

data = rdr.reader("EEG_data.csv")
training_labels = data[['predefinedlabel', 'user-definedlabeln']]
training_data = data.drop(columns = ['predefinedlabel', 'user-definedlabeln'])


# Not using main(), because of Spyder's variable explorer
# If needed, uncomment this part and comment the previous part of code
#def main():
    

#if __name__ == '__main__':
#    main()
