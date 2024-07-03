import torch

class GPU:
    def __init__(self, device, ddp=False):
        self.ddp = ddp
        self.main = device == 0
        self.device = torch.device(device % 2)
        torch.set_default_device(self.device)
        torch.cuda.set_device(self.device)
        print("Using compute module " + str(self.device))