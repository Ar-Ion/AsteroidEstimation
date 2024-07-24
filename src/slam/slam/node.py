import rclpy

import astronet_frontends.factory
from astronet_frontends import AsyncFrontend, DriveClientFrontend

from .backend import SLAMBackend

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("slam_node")
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
        ("config.matcher", rclpy.Parameter.Type.STRING),
        ("config.matcher_args", rclpy.Parameter.Type.STRING_ARRAY),
        ("config.criterion", rclpy.Parameter.Type.STRING),
        ("config.criterion_args", rclpy.Parameter.Type.DOUBLE_ARRAY)
    ])

    size = node.get_parameter("size").value
    mode = node.get_parameter("mode").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))   
    config = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("config").items()))

    frontend_wrapped = astronet_frontends.factory.instance(input_params, mode, size)
    frontend = AsyncFrontend(frontend_wrapped, wait=False, num_workers=1)
    frontend.start()

    slam = SLAMBackend(frontend, size, config)

    try:
        slam.loop()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.stop() 
        
if __name__ == '__main__':
    main()