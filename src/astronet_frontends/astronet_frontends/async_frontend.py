import threading
import multiprocessing
import queue

# Make each AsyncEvent unique and comparable by memory location
class AsyncEvent:
    def __init__(self, uid):
        self._uid = uid
    
    def __eq__(self, other):
        return self._uid == other._uid
        
class AsyncFrontend:
    # Defines the possible events that must be passed between the sync parent process and the async child subprocess
    class Events:
        START = AsyncEvent("start")
        STOP = AsyncEvent("stop")
        FORCE_TICK = AsyncEvent("tick")
        
    # The mode defines if the transmit/receive pipeline should wait until the application requests to send new data
    class Modes:
        WAIT = 1
        NO_WAIT = 2

    def __init__(self, frontend, mode):  
        self._frontend = frontend
        self._mode = mode
        
    def start(self):        
        self._event_queue = multiprocessing.Queue(16)
        self._data_input_queue = multiprocessing.Queue(1024)
        self._data_output_queue = multiprocessing.Queue(1024)
        
        # Defines an async subprocess to speed up the data transmission/reception pipeline.
        # Arguments are passed as simple object and queues to simplify memory management.
        self._process = threading.Thread(
            target=AsyncFrontend.run, 
            args=(
                self._mode,
                self._frontend, 
                self._event_queue, 
                self._data_input_queue, 
                self._data_output_queue,
            )
        )
        
        self._process.start() # Start the actual subprocess
        self._event_queue.put(AsyncFrontend.Events.START) # Send the START signal to the async process
        
    def stop(self):
        self._event_queue.put(AsyncFrontend.Events.STOP) # Send the STOP signal to the async process
        self._process.join() # and wait for it to terminate

    def send_event(self, event):
        self._event_queue.put(event)

    def on_event(frontend, data_input_queue, event):
        if event == AsyncFrontend.Events.START:
            frontend.on_start()
        elif event == AsyncFrontend.Events.STOP:
            frontend.on_stop()
            raise InterruptedError
        elif event == AsyncFrontend.Events.FORCE_TICK:
            data = data_input_queue.get()
            frontend.on_input(data)
        else:
            frontend.on_event(event)
    
    # The AsyncFrontend objects must be designed to have the transmission and reception systems run on a single thread.
    # A packet can only be received if the front-end transmits something, which is a weak assumption for this project
    def run(mode, frontend, event_queue, data_input_queue, data_output_queue):
        def rx_callback(data):
            #print("write " + str(data_output_queue.qsize()))
            data_output_queue.put(data)
        
        frontend.set_receive_callback(rx_callback)
        
        while True:
            # Tick frontend
            frontend.on_tick()

            # Process the event queue
            try:
                is_idle = not frontend.is_running()
                event = event_queue.get(is_idle or mode == AsyncFrontend.Modes.WAIT)
                AsyncFrontend.on_event(frontend, data_input_queue, event)
            except queue.Empty:
                pass # No new event
            except InterruptedError:
                break # Parent process decided to terminate the frontend
                                       
    # Adds data to the queue, so that it will be sent soon by the async subprocess. 
    def transmit(self, data):
        self._data_input_queue.put(data)
        self._event_queue.put(AsyncFrontend.Events.FORCE_TICK)
        
    # Fetches new data from the the async subprocess. May return None if no new data has been received.    
    def receive(self, blocking=False):
        try:
            #print("read " + str(self._data_output_queue.qsize()))
            return self._data_output_queue.get(blocking)
        except queue.Empty:
            return None