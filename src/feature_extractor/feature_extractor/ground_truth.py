class MotionModel:
    def __init__(self):
        self._intrinsics = None
        self._extrinsics = None

    def set_camera_intrinsics(self, intrinsics):
        self._intrinsics = intrinsics

    def set_camera_extrinsics(self, extrinsics):
        self._extrinsics = extrinsics

    def is_ready(self):
        return self._intrinsics != None and self._extrinsics != None
    
    def ensure_ready(self):
        if not self.is_ready():
            raise Exception("Motion model not yet ready. Is the camera_info topic published?")

    def propagate(self):
        self.ensure_ready()

    def camera2object(self, point):
        self.ensure_ready()
        return self._extrinsics.revert(self._intrinsics.revert(point.float()))

    def object2camera(self, point):
        self.ensure_ready()
        return self._intrinsics.apply(self._extrinsics.apply(point.float()))
