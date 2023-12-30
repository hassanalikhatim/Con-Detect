from tensorflow.keras import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from sklearn.utils import shuffle


from _0_general_ML.data_utils.nlp_dataset import NLP_Dataset



class Kaggle_Fakenews(NLP_Dataset):
    
    def __init__(
        self,
        dataset_folder='../../_Datasets/', max_len=100,
        test_ratio=0.17,
        **kwargs
    ):
        
        super().__init__(
            data_name='Kaggle_Fake_News',
            dataset_folder=dataset_folder,
            max_len=max_len, test_ratio=test_ratio
        )
        
        return
    
    
    def load_data(self):
        
        df = pd.read_csv(self.dataset_folder + 'Kaggle_Fake_News/sample2.csv', engine='python', on_bad_lines='skip')
        
        df_train = df[df['type'] == 'training']
        df_train.reset_index(inplace=True)
        xtrain = df_train['content'].tolist()
        ytrain = df_train['label'].tolist()
        
        x_train = []
        y_train = []
        for i in range(len(xtrain)):
            x_train.append(xtrain[i])
            y_train.append([float(ytrain[i]), 1-float(ytrain[i])])
        y_train = np.array(y_train)
        
        df_test = df[df['type'] == 'test']
        df_test.reset_index(inplace=True)
        xtest = df_test['content'].tolist()
        ytest = df_test['label'].tolist()
        
        x_test = []
        y_test = []
        for i in range(len(xtest)):
            x_test.append(xtest[i])
            y_test.append([float(ytest[i]), 1-float(ytest[i])])
        y_test = np.array(y_test)
        
        return x_train, y_train, x_test, y_test