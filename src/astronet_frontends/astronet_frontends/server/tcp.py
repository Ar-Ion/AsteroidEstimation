from .. import Frontend
import socket
import pickle

class TCPServerFrontend(Frontend):
    def __init__(self, ip, port):
        super().__init__()
        self._ip = ip
        self._port = port
        self._connected = False
        self._client = None
        
    def on_start(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind((self._ip, self._port))
        self._server.listen(1)
        print("TCP frontend started")
            
    def on_stop(self):
        self._stop_client()
        self._stop_server()
        print("TCP frontend stopped")
        
    def _stop_server(self):
        if self._server:
            self._server.close()
            
        print("Server stopped")
            
    def _stop_client(self):
        if self._client:
            self._client.close()
            self._connected = False
            
        print("Client disconnected")

    def check_new_connection(self):
        if not self._connected:
            print("Waiting for remote connection...")
            self._client, address = self._server.accept()
            print(str(address) + " is now connected")
            self._connected = True      
                    
    def on_input(self, data):
        try:
            self.check_new_connection()
            
            serialized_data = pickle.dumps(data)

            packet = b'\x7f'
            packet += len(serialized_data).to_bytes(4, byteorder='big')
            packet += serialized_data
            
            self._client.sendall(packet)
        except:
            self._stop_client()
        
        