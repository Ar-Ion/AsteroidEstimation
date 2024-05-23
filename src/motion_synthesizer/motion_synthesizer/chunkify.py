import rclpy
import numpy as np

from astronet_frontends import AsyncFrontend, factory
from .chunkifier import Chunkifier
 
def main(args=None):
    rclpy.init(args=args)
    
    node = rclpy.create_node("motion_synthesizer_node")
    
    node.declare_parameter("size", rclpy.Parameter.Type.INTEGER)
    node.declare_parameter("mode", rclpy.Parameter.Type.STRING)
    node.declare_parameter("input_dim", rclpy.Parameter.Type.INTEGER)
    node.declare_parameter("output_dim", rclpy.Parameter.Type.INTEGER)

    node.declare_parameters("", [
        ("input.type", rclpy.Parameter.Type.STRING),
        ("input.path", rclpy.Parameter.Type.STRING)
    ])
    
    node.declare_parameters("", [
        ("output.type", rclpy.Parameter.Type.STRING),
        ("output.path", rclpy.Parameter.Type.STRING)
    ])
    
    size = node.get_parameter("size").value
    in_dim = node.get_parameter("input_dim").value
    out_dim = node.get_parameter("output_dim").value
    mode = node.get_parameter("mode").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))
    output_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("output").items()))

    chunks_per_row = int(np.ceil(float(in_dim)/float(out_dim)))
    num_chunks = chunks_per_row**2
    
    client_wrapped = factory.instance(input_params, mode, size//2)
    server_wrapped = factory.instance(output_params, mode, size*num_chunks//2) # Each output motion requires two input images
    
    client = AsyncFrontend(client_wrapped, AsyncFrontend.Modes.NO_WAIT)
    server = AsyncFrontend(server_wrapped, AsyncFrontend.Modes.WAIT)

    client.start()
    server.start()
    
    backend = Chunkifier(client, server, size//2, in_dim, out_dim)

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