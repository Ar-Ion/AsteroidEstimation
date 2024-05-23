import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import astronet_frontends.factory
from astronet_frontends import AsyncFrontend, DriveClientFrontend

from .backends import ClassicalMatcher
from .backends import LightglueMatcher
from .backends import Backend
from .backends.metrics import Cosine, L2
from .backends.criteria import GreaterThan, LessThan

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("feature_matcher_node")
    node.declare_parameter("size", rclpy.Parameter.Type.INTEGER)
    node.declare_parameter("mode", rclpy.Parameter.Type.STRING)

    node.declare_parameters("", [
        ("input.type", rclpy.Parameter.Type.STRING),
        ("input.path", rclpy.Parameter.Type.STRING)
    ])

    node.declare_parameters("", [
        ("output.type", rclpy.Parameter.Type.STRING),
        ("output.path", rclpy.Parameter.Type.STRING)
    ])

    size = node.get_parameter("size").value
    mode = node.get_parameter("mode").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))
    output_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("output").items()))

    frontend_wrapped = astronet_frontends.factory.instance(input_params, mode, size)
    frontend = AsyncFrontend(frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    frontend.start()

    matcher = ClassicalMatcher(Cosine, GreaterThan(0.75))
    backend = Backend(frontend, size, matcher)

    try:
        backend.loop()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.stop() 
        
if __name__ == '__main__':
    main()