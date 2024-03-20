class Evaluator:
    def __init__(self, backend):
        self._backend = backend

    # Timestamp in nanoseconds
    def on_input(self, timestamp, image, depth):
        
        kp, des = self._backend.extract_features(image)

        pass