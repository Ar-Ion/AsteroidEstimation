class Frontend:
    
    # No callback is give to the constructor as it is often wrapped in an AsyncFrontend object
    def __init__(self, source, mode, size):
        self.source = source
        self.mode = mode
        self.size = size
        self._rx_callback = None
        
    # Updates the callback function that will be called when we receive new data
    def set_receive_callback(self, cb):
        self._rx_callback = cb
        
    # If returns false, the frontend has stopped running and the thread enters an idle state
    def is_running(self):
        return True
    
    # Called when the frontend is started by the application
    def on_start(self):
        pass
    
    # Called when the frontend is stopped by the application
    def on_stop(self):
        pass

    # Called in an async loop
    def on_tick(self):
        pass
    
    # Called when we need to transmit sth to through this frontend
    def on_input(self, data):
        pass

    # Called when a custom event is sent from the application to the frontend
    def on_event(self, event):
        pass
    
    # Called by the frontend whenever sth is received
    def on_receive(self, data):
        if self._rx_callback != None:
            self._rx_callback(data)