import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from _0_general_ML.data_utils.nlp_dataset import NLP_Dataset



class Whatsapp_Misinformation(NLP_Dataset):
    
    def __init__(
        self,
        dataset_folder='../../_Datasets/', max_len=100,
        test_ratio=0.17,
        **kwargs
    ):
        
        super().__init__(
            data_name='Whatsapp_Misinformation',
            dataset_folder=dataset_folder,
            max_len=max_len, test_ratio=test_ratio
        )
        
        return
    
    
    def load_data(self):
        
        df = pd.read_csv(self.dataset_folder + "Whatsapp_Misinformation/WA_MisInfo_Annotated_Dataset.csv")
        
        x_all = df['data'].tolist()
        y_all = df['labels'].tolist()

        x = []
        y = []
        for i in range(len(y_all)):
            x.append(x_all[i])
            y.append(np.sum('Misinformation' in y_all[i].split(',')))

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15, random_state=42)

        x_train = []
        y_train = []
        for i in range(len(xtrain)):
            x_train.append(xtrain[i])
            y_train.append([float(ytrain[i]), float(1-ytrain[i])]) #0th index is misinformation
        y_train = np.array(y_train)

        x_test = []
        y_test = []
        for i in range(len(xtest)):
            x_test.append(xtest[i])
            y_test.append([float(ytest[i]), float(1-ytest[i])])
        y_test = np.array(y_test)
        
        return x_train, y_train, x_test, y_test