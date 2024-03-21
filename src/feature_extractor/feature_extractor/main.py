import rclpy

from feature_extractor.frontend import ROSFrontend
from feature_extractor.backend import AKAZEBackend
from feature_extractor.evaluator import Evaluator
from feature_extractor.ground_truth import MotionModel

def main(args=None):
    rclpy.init(args=args)

    backend = AKAZEBackend()
    motion_model = MotionModel()
    evaluator = Evaluator(backend, motion_model)
    frontend = ROSFrontend(evaluator, motion_model)

    frontend.loop()
    frontend.cleanup()

    rclpy.shutdown()

if __name__ == '__main__':
    main()