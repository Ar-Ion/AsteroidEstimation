from .. import Frontend
import pickle
import os

class DriveServerFrontend(Frontend):
    def __init__(self, source, mode, size):
        super().__init__(source, mode, size)

        self._current = 0

        self._output_dir = os.path.join(self.source, self.mode)
        os.makedirs(self._output_dir, exist_ok=True)
        
    def on_start(self):
        print("Drive server frontend started")
            
    def on_stop(self):
        print("Drive server frontend stopped")
        
    def is_running(self):
        return self._current < self.size

    def on_input(self, data):
        if self._current < self.size:
            filename = os.path.join(self._output_dir, str(self._current).zfill(6) + ".pickle")
            self._current += 1

            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        