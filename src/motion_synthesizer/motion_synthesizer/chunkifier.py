import torch
from astronet_utils import MotionUtils

class Chunkifier:
    def __init__(self, client, server, size, in_dim, out_dim):
        self._client = client
        self._server = server
        self._size = size
        self._in_dim = in_dim
        self._out_dim = out_dim
        
        # CUDA configuration
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(self._device)
        print("Using compute module " + str(self._device))

    def loop(self):

        count = 0
        
        while count < self._size:
            data_in = self._client.receive(blocking=True)
            data_in_gpu = MotionUtils.to(data_in, device=self._device)
            data_out = MotionUtils.create_chunks(data_in_gpu, self._in_dim, self._out_dim)
            
            for chunk in data_out:
                chunk_cpu = MotionUtils.to(chunk, device="cpu")
                self._server.transmit(chunk_cpu)
                
            count += 1

            if count % 100 == 0:
                print("Chunkified " + f"{count/self._size:.0%}" + " of synthetic motion data")