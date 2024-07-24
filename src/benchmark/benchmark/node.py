import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import astronet_frontends.factory
from astronet_frontends import AsyncFrontend, DriveClientFrontend

from .simple_stats import SimpleStats
from .error_points import ErrorPoints
from .precision_recall import PrecisionRecall
from .visualize import Visualize

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
        ("images.type", rclpy.Parameter.Type.STRING),
        ("images.path", rclpy.Parameter.Type.STRING)
    ])
    
    node.declare_parameters("", [
        ("config.keypoints.metric", rclpy.Parameter.Type.STRING),
        ("config.keypoints.criterion", rclpy.Parameter.Type.STRING),
        ("config.keypoints.criterion_args", rclpy.Parameter.Type.DOUBLE_ARRAY),
        ("config.features.matcher", rclpy.Parameter.Type.STRING),
        ("config.features.matcher_args", rclpy.Parameter.Type.STRING_ARRAY),
        ("config.features.criterion", rclpy.Parameter.Type.STRING),
        ("config.features.criterion_args", rclpy.Parameter.Type.DOUBLE_ARRAY)
    ])

    size = node.get_parameter("size").value
    mode = node.get_parameter("mode").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))
    output_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("output").items()))
    images_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("images").items()))
    config = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("config").items()))

    if output_params["type"] != "Plots":
        raise NotImplementedError("Supported outputs only include 'Plots'")

    frontend_wrapped = astronet_frontends.factory.instance(input_params, mode, size)
    frontend = AsyncFrontend(frontend_wrapped, wait=False, num_workers=1)
    frontend.start()
    
    images_frontend_wrapped = astronet_frontends.factory.instance(images_params, mode, 2*size)
    images_frontend = AsyncFrontend(images_frontend_wrapped, wait=False, num_workers=1)
    images_frontend.start()

    simple_stats = SimpleStats(frontend, size, config)
    error_points = ErrorPoints(frontend, size, config, output_params)
    precision_recall = PrecisionRecall(frontend, size, config, output_params)
    visualize = Visualize(frontend, images_frontend, size, config)

    try:
        simple_stats.loop()       
        #error_points.loop()
        precision_recall.loop()
        visualize.loop() 
        
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.stop() 
        images_frontend.stop() 
        
if __name__ == '__main__':
    main()