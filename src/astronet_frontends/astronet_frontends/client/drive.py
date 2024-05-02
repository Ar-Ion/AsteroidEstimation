from .. import Frontend, AsyncEvent
import pickle
import random
import os

class DriveClientFrontend(Frontend):

    class Events:
        RESET = AsyncEvent("reset")

    def __init__(self, source, mode, size):
        super().__init__(source, mode, size)
        self._indices = list(range(size))
        self._current = 0
        
    def on_start(self):
        print("Drive client frontend started")
            
    def on_stop(self):
        print("Drive client frontend stopped")

    def on_event(self, event):
        # To train on a new epoch, the parent process can reset this frontend by shuffling the dataset and resetting the index
        if event == DriveClientFrontend.Events.RESET:
            random.shuffle(self._indices)
            self._current = 0
            
    def is_running(self):
        return self._current < self.size
    
    def on_input(self):
        raise NotImplementedError()

    def on_tick(self):
        if self._current < self.size:
            id = self._indices[self._current]
            data = self.get(id)
            self._current += 1
            self.on_receive(data)

    def get(self, id):
        filename = os.path.join(self.source, self.mode, str(id).zfill(6) + ".pickle")
        
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            return data