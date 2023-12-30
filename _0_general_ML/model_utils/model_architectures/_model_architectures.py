import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, InputLayer
from tensorflow.keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Reshape


from custom_layer_utils.custom_activation import custom_Activation
from custom_layer_utils.modular_attention import modular_Attention



WEIGHT_DECAY = 0.

def cnn_model(
    data, weight_decay=WEIGHT_DECAY, 
    n_layers=4, activation_name='relu',
    learning_rate=1e-4,
    compile_model=True, apply_softmax=True,
    embedding_depth=30
):
    
    if weight_decay == 0:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(weight_decay)
    
    encoder = Sequential()
    encoder.add(InputLayer(input_shape=data.get_input_shape()))
    
    if data.type == 'nlp':
        encoder.add(tf.keras.layers.Embedding(data.size_of_vocabulary, embedding_depth, input_length=data.max_len))
        encoder.add(tf.keras.layers.Reshape(target_shape=(data.max_len, embedding_depth, 1)))
    
    for l in range(n_layers):
        encoder.add(Conv2D(3, (3,3), padding='same', activity_regularizer=kernel_regularizer))
        encoder.add(custom_Activation(activation_name))
    
    encoder.add(Flatten())
    encoder.add(Dense(data.get_output_shape()[0]))
    
    if apply_softmax:
        encoder.add(Activation('softmax'))
    
    if compile_model:
        encoder.compile(loss='categorical_crossentropy', 
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                        metrics=['accuracy'])
    return encoder


def mlp_model(
    data, weight_decay=WEIGHT_DECAY, 
    n_layers=2, activation_name='square',
    learning_rate=1e-4,
    compile_model=True, apply_softmax=True,
    embedding_depth=30
):
    
    if weight_decay == 0:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(weight_decay)
    
    encoder = Sequential()
    encoder.add(InputLayer(input_shape=data.get_input_shape()))
    
    if data.type == 'nlp':
        encoder.add(tf.keras.layers.Embedding(data.size_of_vocabulary, embedding_depth, input_length=data.max_len))
        encoder.add(tf.keras.layers.Reshape(target_shape=(data.max_len, embedding_depth, 1)))
    
    encoder.add(Flatten())
    
    for l in range(n_layers):
        encoder.add(Dense(20, activity_regularizer=regularizers.l2(weight_decay)))
        encoder.add(custom_Activation(activation_name))
    
    encoder.add(Dense(data.get_output_shape()[0]))
    
    if apply_softmax:
        encoder.add(Activation('softmax'))
    
    if compile_model:
        encoder.compile(
            loss='categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            metrics=['accuracy'])
    
    return encoder


def attention_text(
    data, weight_decay=WEIGHT_DECAY, 
    n_layers=2, activation_name='square',
    learning_rate=1e-4,
    compile_model=True, apply_softmax=True,
    embedding_depth=30
):
    
    if weight_decay == 0:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(weight_decay)
    
    model = tf.keras.Sequential()
    model.add(InputLayer(input_shape=data.get_input_shape()))
    
    if data.type == 'nlp':
        model.add(tf.keras.layers.Embedding(data.size_of_vocabulary, embedding_depth, input_length=data.max_len))
        model.add(tf.keras.layers.Reshape(target_shape=(data.max_len, embedding_depth, 1)))
    
    if "_approx" in activation_name:
        model.add(modular_Attention(30, activation_name='sigmoid_approx', activity_regularizer=kernel_regularizer))
    else:
        model.add(modular_Attention(30, activation_name='sigmoid', activity_regularizer=kernel_regularizer))
    
    model.add(Flatten())
    
    for l in range(n_layers):
        model.add(Dense(30, activity_regularizer=kernel_regularizer))
        model.add(custom_Activation(activation_name))
    
    model.add(Dense(data.get_output_shape()[0]))
    
    if apply_softmax:
        model.add(Activation('softmax'))
    
    if compile_model:
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            metrics=['accuracy'])
    
    return model


def cnn_mini_model(
    data, weight_decay=WEIGHT_DECAY, 
    n_layers=4, activation_name='relu',
    learning_rate=1e-4,
    compile_model=True, apply_softmax=True
):
    
    if weight_decay == 0:
        kernel_regularizer = None
    else:
        kernel_regularizer = regularizers.l2(weight_decay)
    
    encoder = Sequential()
    encoder.add(InputLayer(input_shape=data.get_input_shape()))
    
    encoder.add(tf.keras.layers.Conv2D(16, 8, strides=1, padding='same', activation='relu'))
    encoder.add(tf.keras.layers.MaxPool2D(2, 1))
    encoder.add(tf.keras.layers.Conv2D(32, 4, strides=1, padding='valid', activation='relu'))
    encoder.add(tf.keras.layers.MaxPool2D(2, 1))
    
    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dense(32, activation='relu'))
    
    encoder.add(tf.keras.layers.Dense(data.get_output_shape()[0]))
    
    if apply_softmax:
        encoder.add(Activation('softmax'))
    
    if compile_model:
        encoder.compile(loss='categorical_crossentropy', 
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                        metrics=['accuracy'])
    
    return encoder
    