from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from .models.sparse import SuperPoint
from .dataset import AsteroidMotionDataset
import MinkowskiEngine as ME

class Trainer:

    # Evaluates the performance of a given backend algorithm against a ground-truth motion model
    def __init__(self, train_frontend, validate_frontend, train_size, validate_size):
        
        # Hyperparameters
        self._max_epochs = 1
        lr = 0.01
        momentum = 0.9
        
        # Initialize datasets for training and validate
        train_dataset = AsteroidMotionDataset(train_frontend, train_size)
        validate_dataset = AsteroidMotionDataset(validate_frontend, validate_size)
        
        self._train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, collate_fn=AsteroidMotionDataset.collate)
        self._validate_dataloader = DataLoader(validate_dataset, batch_size=128, shuffle=False, collate_fn=AsteroidMotionDataset.collate)

        # CUDA configuration
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        
        print("Using compute module " + str(self._device))
        
        # Model instantiation
        self._model = SuperPoint().to(self._device)
        self._model.train()
        
        # Optimizer instantiation
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)

    def loss(self, distance_matrix, match_matrix):
        epsilon = 8
        ld = 250
        mp = 1
        mn = 0.2
        
        s = torch.zeros_like(distance_matrix)
        s[distance_matrix < epsilon] = 1
        
        hinge_loss = ld * s * torch.max(torch.tensor(0), mp - match_matrix) + (1 - s) * torch.max(torch.tensor(0), match_matrix - mn)

        #tp = (s * match_matrix).sum()/torch.count_nonzero(s)
        #tn = ((1 - s) * (1 - match_matrix)).sum()/torch.count_nonzero((1 - s))

        #print("True positive: " + str(tp))
        #print("True negative: " + str(tn))

        return hinge_loss

        # normed_match_matrix = match_matrix**2

        # # torch.log(torch.nn.functional.normalize(1 + distance_matrix, dim=0)) +
        # expected_likelihood = torch.log_softmax(-distance_matrix**2/(2*sigma*sigma), dim=0)
        # predicted_likelihood = torch.log(normed_match_matrix / torch.sum(normed_match_matrix, dim=0))

        # kl_div = torch.sum(torch.exp(expected_likelihood) * (expected_likelihood - predicted_likelihood), dim=0)
        
        # return kl_div.mean()
        
    def to_device(self, sample):
        c1, c2, f1, f2 = sample
        
        c1_dev = c1.to(self._device)
        c2_dev = c2.to(self._device)
        f1_dev = f1.to(self._device)
        f2_dev = f2.to(self._device)
    
        return (c1_dev, c2_dev, f1_dev, f2_dev)
    
    def forward(self, sample):
        # "i" stands for "input", "o" stands for "output", "c" stands for "coordinate", "f" stands for "feature" 
        (ic1, ic2, if1, if2) = sample

        in1 = ME.SparseTensor(if1[:, None], ic1[:, 0:3])
        in2 = ME.SparseTensor(if2[:, None], ic2)
        
        out1 = self._model.forward(in1)
        out2 = self._model.forward(in2)
        
        oc1 = out1.coordinates
        of1 = out1.features
        oc2 = out2.coordinates
        of2 = out2.features
        
        return (oc1, oc2, of1, of2)
        
    # L2 metric
    def true_distance(self, sample):
        (c1, c2, f1, f2) = sample
        return torch.cdist(c1.float(), c2.float())
        
    # Cosine-similarity metric
    def predicted_distance(self, sample):
        (c1, c2, f1, f2) = sample
        return f1 @ f2.T

    def loop(self):
        for epoch in range(self._max_epochs):
            # Training
            iter = 0
            
            for sample in self._train_dataloader:
                sample_gpu = self.to_device(sample)
                model_output = self.forward(sample_gpu)
                true_distance = self.true_distance(model_output)
                predicted_distance = self.predicted_distance(model_output)
                                            
                loss = self.loss(true_distance, predicted_distance)
                avg_loss = loss.mean()
                avg_loss.backward()
                
                self._optimizer.step()
                self._optimizer.zero_grad()
                
                if iter % 10 == 0:
                    print(f"Iteration {iter} on epoch {epoch} finished with loss {avg_loss}")
                    
                iter += 1
            
            losses = []
            tp = []
            tn = []
            
            # Validation
            with torch.set_grad_enabled(False):
                for sample in self._train_dataloader:
                    sample_gpu = self.to_device(sample)
                    model_output = self.forward(sample_gpu)
                    true_distance = self.true_distance(model_output)
                    predicted_distance = self.predicted_distance(model_output)
                    
                    epsilon = 8
                    s = torch.zeros_like(true_distance)
                    s[true_distance < epsilon] = 1
                    
                    losses.append(self.loss(true_distance, predicted_distance).mean())
                    tp.append((s * predicted_distance).sum()/torch.count_nonzero(s))
                    tn.append(((1 - s) * (1 - predicted_distance)).sum()/torch.count_nonzero((1 - s)))
                    
            avg_loss = torch.tensor(losses).mean()
            avg_tp = torch.tensor(tp).mean()
            avg_tn = torch.tensor(tn).mean()

            print(f"Epoch {epoch} finished with loss {avg_loss}, true positives {avg_tp}, true_negatives {avg_tn}")
            