from .. import Frontend
import pickle
import os

class DriveServerFrontend(Frontend):
    def __init__(self, source, mode, size):
        super().__init__(source, mode, size)

        self._output_dir = os.path.join(self.source, self.mode)
        os.makedirs(self._output_dir, exist_ok=True)
        
    def on_start(self):
        print("Drive server frontend started")
            
    def on_stop(self):
        print("Drive server frontend stopped")

    def on_input(self, input):
        resource_id, data = input
        id = resource_id % self.size
        
        filename = os.path.join(self._output_dir, str(id).zfill(6) + ".pickle")

        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        