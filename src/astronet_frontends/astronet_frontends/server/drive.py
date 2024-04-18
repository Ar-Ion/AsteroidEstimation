from .. import Frontend
import pickle
import os

class DriveServerFrontend(Frontend):
    def __init__(self, folder, size):
        super().__init__()
        self._folder = folder
        self._size = size
        self._current = 0
        
    def on_start(self):
        print("Drive server frontend started")
            
    def on_stop(self):
        print("Drive server frontend stopped")
        
    def is_running(self):
        return self._current < self._size

    def on_input(self, data):
        if self._current < self._size:
            filename = os.path.join(self._folder, str(self._current).zfill(6) + ".pickle")
            self._current += 1

            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        