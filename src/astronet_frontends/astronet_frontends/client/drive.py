from .. import Frontend, AsyncEvent
import pickle
import random
import os

class DriveClientFrontend(Frontend):

    class Events:
        RESET = AsyncEvent("reset")

    def __init__(self, folder, size):
        super().__init__()
        self._folder = folder
        self._size = size
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
        return self._current < self._size

    def on_tick(self):
        if self._current < self._size:
            id = self._indices[self._current]
            filename = os.path.join(self._folder, str(id).zfill(6) + ".pickle")
            self._current += 1

            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
                self.receive(data)