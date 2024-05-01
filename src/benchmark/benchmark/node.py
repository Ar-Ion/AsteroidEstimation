import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import astronet_frontends.factory
from astronet_frontends import AsyncFrontend

from .benchmarker import Benchmarker

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("benchmark_node")
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
    
    node.declare_parameters("", [
        ("config.coords.metric", rclpy.Parameter.Type.STRING),
        ("config.coords.criterion", rclpy.Parameter.Type.STRING),
        ("config.coords.criterion_args", rclpy.Parameter.Type.DOUBLE_ARRAY),
        ("config.features.metric", rclpy.Parameter.Type.STRING),
        ("config.features.criterion", rclpy.Parameter.Type.STRING),
        ("config.features.criterion_args", rclpy.Parameter.Type.DOUBLE_ARRAY)
    ])

    size = node.get_parameter("size").value
    mode = node.get_parameter("mode").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))
    output_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("output").items()))
    config = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("config").items()))

    if output_params["type"] != "Plots":
        raise NotImplementedError("Supported outputs only include 'Plots'")

    frontend_wrapped = astronet_frontends.factory.instance(input_params, mode, size)
    frontend = AsyncFrontend(frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    frontend.start()

    benchmarker = Benchmarker(frontend, size, config)

    try:
        benchmarker.loop()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.stop() 
        
if __name__ == '__main__':
    main()