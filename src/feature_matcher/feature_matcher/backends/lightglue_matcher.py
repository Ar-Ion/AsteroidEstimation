import torch
from coffee_nn.models.matchers import LightGlue
from .matcher import Matcher

class LightglueMatcher(Matcher):
    def __init__(self, model_path, criterion=None, autoload=True):
        super().__init__(criterion)

        self._model_path = model_path
        
        # Load GPU
        print("Loading GPU...")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("GPU loaded. Using compute device: " + str(self._device))
        
        # Model instantiation
        model_wrapped = LightGlue(features="coffee")
        self.model = model_wrapped.float().to(self._device)
        self.model.eval()
        
        if autoload:
            print("Loading matcher model checkpoint...")
            self.load()
            
    def load(self):
        self.model.load_state_dict(torch.load(self._model_path))
        
    def save(self):
        torch.save(self.model.state_dict(), self._model_path)
        
    def match(self, data):
        
        size = torch.tensor((1024, 1024))

        a_data = {
            "keypoints": data.prev_kps[None, :, 1:3], 
            "descriptors": data.prev_features[None, :], 
            "image_size": size
        }
        
        b_data = {
            "keypoints": data.next_kps[None, :, 1:3], 
            "descriptors": data.next_features[None, :], 
            "image_size": size
        }

        output = self.model({
            "image0": a_data, 
            "image1": b_data
        })
        
        scores = output["scores"][0]
        
        pred_dists = scores[:-1, :-1].exp() # Remove dustbin and convert log-score to score
        pred_matches = self._criterion.apply(pred_dists)
                
        return pred_dists, pred_matches