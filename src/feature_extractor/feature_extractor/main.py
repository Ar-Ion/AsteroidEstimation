import rclpy

from feature_extractor.frontend import ROSFrontend, TCPFrontend
from feature_extractor.backend import COFFEE_V2_Backend
from feature_extractor.backend import SIFTBackend
from feature_extractor.backend import ORBBackend

from feature_extractor.evaluator import Evaluator
from feature_extractor.trainer import Trainer
from feature_extractor.ground_truth import MotionModel

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