from .. import Frontend, AsyncEvent
import pickle
import random
import os

class DriveClientFrontend(Frontend):

    class Events:
        RESET = AsyncEvent("reset")

    def __init__(self, source, mode, size):
        super().__init__(source, mode, size)
        
    def on_start(self):
        print("Drive client frontend started")
            
    def on_stop(self):
        print("Drive client frontend stopped")

    def on_input(self, input):
        resource_id, _ = input
        id = resource_id % self.size
        data = self.get(id)
        self.on_receive((id, data))

    def get(self, id):
        filename = os.path.join(self.source, self.mode, str(id).zfill(6) + ".pickle")
        
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            return data