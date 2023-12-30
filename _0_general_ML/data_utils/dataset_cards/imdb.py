import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf


from _0_general_ML.data_utils.nlp_dataset import NLP_Dataset



class IMDB(NLP_Dataset):
    
    def __init__(
        self,
        data_name='IMDB',
        dataset_folder='../../_Datasets/',
        max_len=100, test_ratio=0.17,
        **kwargs
    ):
        
        super().__init__(
            data_name,
            dataset_folder=dataset_folder,
            max_len=max_len)
        
        return
    
    
    def load_data(self):
        
        train_df = pd.read_csv(self.dataset_folder + 'IMDB/train.csv')
        ytrain = train_df['label'].tolist()
        x_train = train_df['content'].tolist()
        
        test_df = pd.read_csv(self.dataset_folder + 'IMDB/test.csv')
        ytest = test_df['label'].tolist()
        x_test = test_df['content'].tolist()
        
        num_classes = len(list(set(ytrain)))
        
        y_train = tf.keras.utils.to_categorical(np.array(ytrain)-1, num_classes)
        y_test = tf.keras.utils.to_categorical(np.array(ytest)-1, num_classes)
        
        return x_train, y_train, x_test, y_test
    
    