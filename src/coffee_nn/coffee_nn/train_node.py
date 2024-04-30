import rclpy

from astronet_frontends import AsyncFrontend, factory
from coffee_nn.trainer import Trainer

def main(args=None):
    rclpy.init(args=args)
    
    node = rclpy.create_node("coffee_nn_train_node")
    
    node.declare_parameter("train_size", rclpy.Parameter.Type.INTEGER)
    node.declare_parameter("validate_size", rclpy.Parameter.Type.INTEGER)
    node.declare_parameter("iter_ratio", rclpy.Parameter.Type.INTEGER)

    node.declare_parameters("", [
        ("input.type", rclpy.Parameter.Type.STRING),
        ("input.path", rclpy.Parameter.Type.STRING)
    ])
    
    node.declare_parameters("", [
        ("output.model_type", rclpy.Parameter.Type.STRING),
        ("output.model_path", rclpy.Parameter.Type.STRING)
    ])
    
    train_size = node.get_parameter("train_size").value
    validate_size = node.get_parameter("validate_size").value
    iter_ratio = node.get_parameter("iter_ratio").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))
    output_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("output").items()))

    train_iter_size = int(train_size / iter_ratio)
    validate_iter_size = int(validate_size / iter_ratio)
    
    train_frontend_wrapped = factory.instance(input_params, "train", train_size)
    train_frontend = AsyncFrontend(train_frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    train_frontend.start()
    
    validate_frontend_wrapped = factory.instance(input_params, "validate", validate_size)
    validate_frontend = AsyncFrontend(validate_frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    validate_frontend.start()
    
    backend = Trainer(train_frontend, validate_frontend, train_iter_size, validate_iter_size, output_params)

    try:
        backend.loop()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        train_frontend.stop()
        validate_frontend.stop()
        
if __name__ == '__main__':
    main()