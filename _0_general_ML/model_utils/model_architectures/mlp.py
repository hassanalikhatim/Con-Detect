import tensorflow as tf


from _0_general_ML.data_utils.dataset import Dataset



WEIGHT_DECAY = 1e-2


def mlp(
    data: Dataset, model_configuration: dict,
    **kwargs
):
    
    default_model_configuration = {
        'weight_decay': WEIGHT_DECAY,
        'learning_rate': 1e-4,
        'embedding_depth': 30,
        'embedding_matrix': None,
        'apply_softmax': True,
        'compile_model': True
    }
    for key in model_configuration.keys():
        default_model_configuration[key] = model_configuration[key]
    
    if default_model_configuration['weight_decay'] == 0:
        kernel_regularizer = None
    else:
        kernel_regularizer = tf.keras.regularizers.l2(
            default_model_configuration['weight_decay']
        )
    
    model = tf.keras.Sequential()
    
    # embedding layer
    if default_model_configuration['embedding_matrix']:
        model.add(
            tf.keras.layers.Embedding(
                data.size_of_vocabulary, default_model_configuration['embedding_depth'], 
                weights=[default_model_configuration['embedding_matrix']], 
                input_length=data.max_len,
                trainable=True
            )
        )
    else:
        model.add(
            tf.keras.layers.Embedding(
                data.size_of_vocabulary, default_model_configuration['embedding_depth'], 
                input_length=data.max_len
            )
        )
    
    model.add(tf.keras.layers.Dropout(0.8))
    model.add(tf.keras.layers.Flatten())
    
    model.add(
        tf.keras.layers.Dense(
            20, activation='relu', 
            activity_regularizer=kernel_regularizer
        )
    )
    
    model.add(
        tf.keras.layers.Dense(
            data.get_output_shape()[0], 
            kernel_regularizer=kernel_regularizer
        )
    )
    
    if default_model_configuration['apply_softmax']:
        model.add(tf.keras.layers.Activation('softmax'))
    
    if default_model_configuration['compile_model']:
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=default_model_configuration['learning_rate']
            ), 
            metrics=['accuracy']
        )
    
    return model