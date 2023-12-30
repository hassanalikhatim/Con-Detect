from textattack.models.wrappers import ModelWrapper


from _0_general_ML.model_utils.model import Keras_Model



class CustomKerasModelWrapper(ModelWrapper):
    
    def __init__(
        self, model: Keras_Model
    ):
        
        self.model = model
        
        return
    
 
    def __call__(self, text_input_list):
        
        prediction = self.model.call(
            self.model.data.get_tokenized(text_input_list))
        
        return [list(prediction[i]) for i in range(0, len(prediction))]