import torch
from coffee_nn.models.descriptors import SparseSuperPoint

class COFFEEDescriptor:
    def __init__(self, model_path, autoload=True):
        self._model_path = model_path
                
        # Load GPU 
        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))
        
        # Model instantiation
        model_wrapped = SparseSuperPoint()
        self.model = model_wrapped.to(self._device)
        self.model.eval()
        
        if autoload:
            print("Loading descriptor model checkpoint...")
            self.load()
            
    def load(self):
        self.model.load_state_dict(torch.load(self._model_path))
        
    def save(self):
        torch.save(self.model.state_dict(), self._model_path)
        