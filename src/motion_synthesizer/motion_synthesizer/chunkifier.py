from astronet_utils import MotionUtils

class Chunkifier:
    def __init__(self, client, server, size, in_dim, out_dim):
        self._client = client
        self._server = server
        self._size = size
        self._in_dim = in_dim
        self._out_dim = out_dim

    def loop(self):

        count = 0
        
        while count < self._size:
            data_in = self._client.receive(blocking=True)
            data_out = MotionUtils.create_chunks(data_in, self._in_dim, self._out_dim)
            
            for chunk in data_out:
                self._server.transmit(chunk)
                
            count += 1

            if count % 100 == 0:
                print("Chunkified " + f"{count/self._output_size:.0%}" + " of synthetic motion data")