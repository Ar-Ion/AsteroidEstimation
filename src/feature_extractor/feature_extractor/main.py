import rclpy

from feature_extractor.frontend import ROSFrontend
from feature_extractor.backend import ORBBackend
from feature_extractor.evaluator import Evaluator

def main(args=None):
    rclpy.init(args=args)

    backend = ORBBackend()
    evaluator = Evaluator(backend)
    frontend = ROSFrontend(evaluator)

    frontend.loop()
    frontend.cleanup()

    rclpy.shutdown()

if __name__ == '__main__':
    main()