import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

class ErrorStatistics:
    def __init__(self):
        self._lengths = []
        self._stddevs = []
        self._count = 0
    
    def add_trajectory(self, keypoints, errors):
        if len(errors) > 0:

            # plt.figure()
            # plt.plot(keypoints)
            # plt.plot(keypoints + errors)
            # plt.show()

            self._lengths.append(len(errors))
            self._stddevs.append(int(100*np.linalg.norm(errors, ord=2)/np.sqrt(len(errors)))/100.0)
            
            self._count += 1
                    
    def display_statistics(self):
        print("Average length: " + str(np.mean(self._lengths)))
        print(np.median(self._stddevs))

        i = 0
        while os.path.exists("data%s.csv" % i):
            i += 1

        data = np.vstack((self._lengths, self._stddevs))
        df = pd.DataFrame(data.T)
        df.to_csv("data%s.csv" % i, index=False)

        plt.hist(self._stddevs, bins=np.linspace(0, 2*np.median(self._stddevs), int(np.sqrt(self._count))))
        plt.show()