import numpy as np
from textattack.attack_recipes import PWWSRen2019, TextBuggerLi2018 
from textattack.attack_recipes import TextFoolerJin2019, DeepWordBugGao2018
from textattack.attack_recipes import BAEGarg2019, BERTAttackLi2020


from _1_nlp_attack.text_attack_utils.model_wrapper_for_attack import CustomKerasModelWrapper



class Text_Attack:
    
    def __init__(
        self, model, attack_inputs
    ):
        
        self.model = CustomKerasModelWrapper(model)
        self.attack_inputs = attack_inputs
        
        self.all_supported_attacks = {
            'text_bugger': TextBuggerLi2018,
            'text_fooler': TextFoolerJin2019,
            'pwws': PWWSRen2019,
            'bae': BAEGarg2019
        }
        
        self.built_attacks = {}
        
        return
    
    
    def build_attack(self, attack_name):
        
        self.built_attacks[attack_name] = self.all_supported_attacks[attack_name].build(self.model)
        
        return
    
    
    def attack(
        self, attack_name
    ):
        
        print('\rStep 1: Building the attack.', end='')
        if attack_name not in self.built_attacks.keys():
            self.build_attack(attack_name)
            
        print('\rStep 2: Generating the attack.', end='')
        results_iterable = self.built_attacks[attack_name].attack_dataset(
            self.attack_inputs, 
            np.arange(len(self.attack_inputs))
        )
        
        return
    
    