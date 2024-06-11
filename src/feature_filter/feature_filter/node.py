import rclpy

import astronet_frontends.factory
import feature_filter.backends.factory
from astronet_frontends import AsyncFrontend

def main(args=None):
    rclpy.init(args=args)
    
    node = rclpy.create_node("feature_filter_node")
    
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
        ("filter_config.backend", rclpy.Parameter.Type.STRING),
        ("filter_config.model_type", rclpy.Parameter.Type.STRING),
        ("filter_config.model_path", rclpy.Parameter.Type.STRING),
        ("filter_config.autoload", rclpy.Parameter.Type.BOOL)
    ])

    size = node.get_parameter("size").value
    mode = node.get_parameter("mode").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))
    output_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("output").items()))
    filter_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("filter_config").items()))
    
    client_wrapped = astronet_frontends.factory.instance(input_params, mode, size)
    server_wrapped = astronet_frontends.factory.instance(output_params, mode, size)

    client = AsyncFrontend(client_wrapped, wait=False, num_workers=1) # Preserve input order by using a single worker (slower though)
    server = AsyncFrontend(server_wrapped)

    client.start()
    server.start()
    
    backend = feature_filter.backends.factory.instance(filter_params, client, server, size)

    try:
        backend.loop()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        server.stop()  
    
if __name__ == '__main__':
    main()