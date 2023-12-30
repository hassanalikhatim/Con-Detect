import tensorflow as tf
import numpy as np


from _2_condetect.defended_model_wrappers.condetect import Con_Detect



class Word_Substitution(Con_Detect):
    """
    This model wrapper is not differentiable, and therefore can only be used for black-box
    adversarial attacks.
    You can make this wrapper differentiable by appropriately modifying it, converting it into
    a tensorlfow layer and placing it after the embedding layer. 
    """
    
    def __init__(
        self, model
    ):
        
        super().__init__()
        
        self.model = model
        
        self.threshold = 0.5
        self.unknown_token = 0
        
        return
    
    
    def modify_input(self, input_, i):
        
        modified_input = input_.copy()
        modified_input[i] = self.unknown_token
        
        return modified_input
    