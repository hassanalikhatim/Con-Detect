import tensorflow as tf
import numpy as np



class Con_Detect:
    """
    This model wrapper is not differentiable, and therefore can only be used for black-box
    adversarial attacks.
    You can make this wrapper differentiable by appropriately modifying it, converting it into
    a tensorlfow layer and placing it after the embedding layer. 
    """
    
    def __init__(
        self, model
    ):
        
        self.model = model
        
        self.threshold = 0.5
        self.allowed_false_positives = 0.15
        
        return
    
 
    def defended_call(self, inputs):
        """
        {inputs}: {inputs} are expected to be the inputs to the first layer of the {self.model}
            with dimension {None(batch_size)} x {max_len}, where {max_len} is the length
            of the sequence.
        """
        
        batch_size, len_of_input = inputs.shape
        
        actual_outputs = self.model.predict(inputs, verbose=False)
        actual_outputs = np.expand_dims(actual_outputs, axis=0)
        
        modified_predictions = np.zeros( [len_of_input] + list(actual_outputs.shape[1:]) )
        for _in, input_ in enumerate(inputs):
            modified_input = np.zeros( (len_of_input, len_of_input) )
            for i in range(len_of_input):
                modified_input[i] = self.modify_input(input_, i)
                
            modified_predictions[:, _in] = self.model.predict(modified_input, verbose=False)
        
        cumulative_contribution_scores = np.mean(np.abs(actual_outputs - modified_predictions), axis=(0,2))
        assert len(cumulative_contribution_scores.shape) == 1, 'Contribution scores should be like list.'
        assert len(cumulative_contribution_scores) == batch_size, 'Contribution scores should be equal to batch size.'
        
        adversarial_probabilities = 1/(1 + np.exp(self.threshold - cumulative_contribution_scores))
        
        outputs = {
            'predictions': actual_outputs[0],
            'adversarial_probabilities': adversarial_probabilities,
            'cumulative_contribution_scores': cumulative_contribution_scores
        }
        
        return outputs
    
    
    def prepare_threshold(self, clean_inputs):
        """
        This function heuristically prepares the threshold based on the clean inputs such that
        the number of {clean_inputs} marked as adversarial are less than {self.allowed_false_positives}.
        
        Inputs:
            {clean_inputs}: Clean inputs that are unperturbed by any attack expected to be the input
                of the first layer of the model.
        """
        
        min_threshold, max_threshold = 0., 1.
        
        cumulative_contribution_scores = self.defended_call(clean_inputs)['cumulative_contribution_scores']
        
        while (max_threshold - min_threshold) >= 0.01:
            self.threshold = (min_threshold + max_threshold)/2
            
            false_positives = np.mean( cumulative_contribution_scores > self.threshold )
            if false_positives > self.allowed_false_positives:
                min_threshold = self.threshold
            else:
                max_threshold = self.threshold
        
        return
    
    
    def modify_input(self, input_, i):
        
        print(
            '\n\nWARNING: This is a general Con-Detect wrapper '
            'and does not implement a specific contribution score function. '
            'Please call the specific subclass of the wrapper to implement defense.'
        )
        
        return input_
    
    
    def call(self, inputs):
        return self.model(inputs)
    
    