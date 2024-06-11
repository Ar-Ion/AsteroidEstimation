import torch

class GPU:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self.device)
        print("Using compute module " + str(self.device))