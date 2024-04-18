class CameraProjection:
    def __init__(self):
        self.__init__(None, None)

    def __init__(self, intrinsics, extrinsics):
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics

    def set_camera_intrinsics(self, intrinsics):
        self._intrinsics = intrinsics

    def set_camera_extrinsics(self, extrinsics):
        self._extrinsics = extrinsics

    def is_ready(self):
        return self._intrinsics != None and self._extrinsics != None
    
    def ensure_ready(self):
        if not self.is_ready():
            raise Exception("Motion model not yet ready. Please first set the intrinsic and extrinsic matrices")

    def camera2object(self, point):
        self.ensure_ready()
        return self._extrinsics.revert(self._intrinsics.revert(point.float()))

    def object2camera(self, point):
        self.ensure_ready()
        return self._intrinsics.apply(self._extrinsics.apply(point.float()))
