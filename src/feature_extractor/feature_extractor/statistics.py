import numpy as np
from matplotlib import pyplot as plt

class ErrorStatistics:
    def __init__(self):
        self._lengths = []
        self._stddevs = []
        self._count = 0
    
    def add_trajectory(self, keypoints, errors):
        if len(errors) > 0:
            self._lengths.append(len(keypoints))
            self._stddevs.append(int(100*np.std(errors))/100.0)
            
            self._count += 1
            
            self.display_statistics()
        
    def display_statistics(self):
        
        print("Valid trajectories: " + str(self._count))
        
        if self._count % 10000 == 0:
            plt.hist(self._stddevs, bins=np.linspace(0, np.mean(self._stddevs) * 3, int(np.sqrt(self._count))))
            plt.show()