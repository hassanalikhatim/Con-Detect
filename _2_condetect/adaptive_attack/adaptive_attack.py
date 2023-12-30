import tensorflow as tf
import numpy as np



class Model_Wrapper_for_Adpative_Attack:
    
    def __init__(
        self, defended_model
    ):
        
        self.defended_model = defended_model
        
        return
    
    
    def prepare_wrapper(self, unperturbed_inputs):
        
        actual_predictions = self.defended_model.model.predict(unperturbed_inputs)
        
        self.hard_actual_predictions = np.zeros_like(actual_predictions)
        self.hard_actual_predictions[
            np.arange(len(actual_predictions)), 
            np.argmax(actual_predictions, axis=1)] = 1.
        
        return
    
    
    def call(self, inputs):
        
        outputs = self.defended_model.defended_call(inputs)
        
        predictions = outputs['predictions']
        adversarial_probabilities = outputs['adversarial_probabilities']
        
        return (1 - adversarial_probabilities) * predictions + adversarial_probabilities * self.hard_actual_predictions
        
        