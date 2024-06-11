import threading
import multiprocessing
import queue
import ctypes

# Make each AsyncEvent unique and comparable by memory location
class AsyncEvent:
    def __init__(self, uid):
        self._uid = uid
    
    def __eq__(self, other):
        return self._uid == other._uid
        
class AsyncFrontend:

    def __init__(self, frontend, num_workers=16, wait=True, random=False):  
        self._frontend = frontend
        self._wait = wait
        
        self._resource_counter = multiprocessing.Value(ctypes.c_longlong, 0)
        self._alive = multiprocessing.Value(ctypes.c_bool, True)    
        self._input_queue = multiprocessing.Queue(512)
        self._output_queue = multiprocessing.Queue(512)
        
        # Add affine coefficients for random number generation
        if random:
            # Large (co-)prime numbers (similar to Java implementation for RNG)
            self._multiplier = 193939
            self._offset = 199933
        else:
            self._multiplier = 1
            self._offset = 1
        
        # Defines an async subprocess to speed up the data transmission/reception pipeline.
        # Arguments are passed as simple object and queues to simplify memory management.
        self._processes = []
        
        for i in range(num_workers):
            self._processes.append(
                multiprocessing.Process(
                    target=AsyncFrontend.run_process, 
                    args=(
                        self._alive,
                        self._frontend, 
                        self._input_queue, 
                        self._output_queue,
                    )
                )
            )
        
        self._scheduler = threading.Thread(
            target=AsyncFrontend.run_scheduler, 
            args=(
                self._alive,
                self._resource_counter, 
                self._input_queue,
                self._multiplier, 
                self._offset
            )
        )
        
    def start(self):
        for process in self._processes:
            process.start()
        
        if not self._wait:
            self._scheduler.start()
        
    def stop(self):
        
        self._alive.value = False
                
        # Clear the input queue once, to allow the scheduler to terminate
        while True:
            try:
                self._input_queue.get(timeout=1)
            except queue.Empty:
                break

        # Stop the scheduler if active, so that nothing else gets added to the input queue
        if not self._wait:
            self._scheduler.join()

        # Wait for subprocesses to terminate
        for process in self._processes:
            process.kill()
            
                            
    # The AsyncFrontend objects must be designed to have the transmission and reception systems run on a single thread.
    # A packet can only be received if the front-end transmits something, which is a weak assumption for this project
    def run_process(alive, frontend, input_queue, output_queue):
        def rx_callback(data):
            output_queue.put(data)

        frontend.set_receive_callback(rx_callback)
        
        while alive.value:
            # Process the event queue
            try:
                input = input_queue.get()
                
                if input != None:
                    frontend.on_input(input)
            except KeyboardInterrupt:
                break # Parent process decided to terminate the frontend
                        
    def run_scheduler(alive, resource_counter, input_queue, multiplier, offset):                
        while alive.value:
            try:
                with resource_counter.get_lock():
                    input_queue.put((resource_counter.value, None))
                    resource_counter.value = resource_counter.value * multiplier + offset
            except KeyboardInterrupt:
                break # Parent process decided to terminate the frontend
                                       
    # Adds data to the queue, so that it will be sent soon by the async subprocess. 
    def transmit(self, data):
        with self._resource_counter.get_lock():
            self._input_queue.put((self._resource_counter.value, data))
            self._resource_counter.value = self._resource_counter.value * self._multiplier + self._offset
        
    # Fetches new data from the the async subprocess. May return None if no new data has been received.    
    def receive(self, blocking=False):
        try:
            (id, data) = self._output_queue.get(blocking)
            return data
        except queue.Empty:
            return None
                
    def get_sync(self, index):
        raise NotImplementedError()