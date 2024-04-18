class FeatureData:
    def __init__(self, env_data, robot_data):
        self.env_data = env_data
        self.robot_data = robot_data

    class EnvironmentData:
        def __init__(self, pose):
            self.pose = pose        

    class RobotData:
        def __init__(self, cam_data):
            self.cam_data = cam_data

        class SparseCameraData:
            def __init__(self, pose, k, coords, features):
                self.pose = pose
                self.k = k
                self.coords = coords
                self.features = features