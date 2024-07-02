import torch

class GPU:
    def __init__(self, device, ddp=False):
        self.ddp = ddp
        self.device = torch.device(device)
        torch.set_default_device(self.device)
        torch.cuda.set_device(self.device)
        print("Using compute module " + str(self.device))