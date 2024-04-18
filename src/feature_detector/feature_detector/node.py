import rclpy

from feature_descriptor.frontend import ROSFrontend, TCPFrontend
from feature_descriptor.backend import COFFEE_V2_Backend
from feature_descriptor.backend import SIFTBackend
from feature_descriptor.backend import ORBBackend

from feature_descriptor.evaluator import Evaluator
from feature_descriptor.trainer import Trainer
from feature_descriptor.ground_truth import MotionModel

def main(args=None):
    rclpy.init(args=args)

    backend = COFFEE_V2_Backend()
    motion_model = MotionModel()
    evaluator = Trainer(backend, motion_model)
    frontend = TCPFrontend(evaluator, motion_model)

    frontend.loop()
    frontend.cleanup()

    rclpy.shutdown()

if __name__ == '__main__':
    main()