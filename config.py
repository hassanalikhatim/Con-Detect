# Hybrid configurations
kaggle_fakenews = {
   'epochs': 1000,
    'batch_size': 256,
    'experiment_repititions': 10,
    'perturbations': [0.02], #0.01, 0.02, 0.05, 0.1],
    'ensemble': 500,
    'patience': 40,
    'learning_rate': 1e-3,
    'max_len': 100
}


general_configurations = {
    'Kaggle_Fake_News': kaggle_fakenews,
    'IMDB': kaggle_fakenews
}


# Changeable configurations
# General
path = '__all_results/results_1'


# Data
dataset_folder = '../../_Datasets/'

dataset_names = []
dataset_names += ['Kaggle_Fake_News']
dataset_names += ['IMDB']

n_samples = 90


# Model
embedding_depth = 10

model_architectures = []
model_architectures += ['mlp']
model_architectures += ['cnn']
model_architectures += ['lstm']
model_architectures += ['hybrid_cnn_rnn']

