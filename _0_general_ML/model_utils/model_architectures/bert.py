import tensorflow as tf
import tensorflow_hub as hub
from official.nlp import optimization


from _0_general_ML.data_utils.dataset import Dataset



WEIGHT_DECAY = 1e-2


def bert(
    data: Dataset, model_configuration: dict,
    **kwargs
):
    
    default_model_configuration = {
        'weight_decay': WEIGHT_DECAY,
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
        
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
    
    input_word_ids = tf.keras.layers.Input(
        shape=(data.max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(
        shape=(data.max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(
        shape=(data.max_len,), dtype=tf.int32, name="input_type_ids")
    
    input_dict = {'input_mask':input_mask, 'input_type_ids':segment_ids, 'input_word_ids':input_word_ids}
    
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    
    outputs = encoder(input_dict)
    net = outputs['pooled_output']  
    
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(
        data.get_output_shape()[0], activation='softmax', name='classifier')(net)
    
    classifier_model = tf.keras.Model(
        inputs=[input_mask, segment_ids, input_word_ids], 
        outputs=net)
    # tf.keras.utils.plot_model(classifier_model)
    
    epochs = 5
    steps_per_epoch = 625 #tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = optimization.create_optimizer(
        init_lr=3e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw')

    if default_model_configuration['compile_model']:
        classifier_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=tf.metrics.BinaryAccuracy())
    
    return classifier_model