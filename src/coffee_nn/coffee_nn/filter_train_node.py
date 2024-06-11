import rclpy

from astronet_frontends import AsyncFrontend, factory
from .filter_trainer import FilterTrainer

def main(args=None):
    rclpy.init(args=args)
    
    node = rclpy.create_node("coffee_nn_filter_train_node")
    
    node.declare_parameter("train_size", rclpy.Parameter.Type.INTEGER)
    node.declare_parameter("validate_size", rclpy.Parameter.Type.INTEGER)

    node.declare_parameters("", [
        ("input.type", rclpy.Parameter.Type.STRING),
        ("input.path", rclpy.Parameter.Type.STRING)
    ])
    
    node.declare_parameters("", [
        ("filter_config.model_path", rclpy.Parameter.Type.STRING),
    ])
    
    train_size = node.get_parameter("train_size").value
    validate_size = node.get_parameter("validate_size").value
    input_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("input").items()))
    filter_params = dict(map(lambda x: (x[0], x[1].value), node.get_parameters_by_prefix("filter_config").items()))

    train_frontend_wrapped = factory.instance(input_params, "train", train_size)
    train_frontend = AsyncFrontend(train_frontend_wrapped, wait=False, random=True)
    train_frontend.start()
    
    validate_frontend_wrapped = factory.instance(input_params, "validate", validate_size)
    validate_frontend = AsyncFrontend(validate_frontend_wrapped, wait=False, random=True)
    validate_frontend.start()
    
    backend = FilterTrainer(train_frontend, validate_frontend, int(train_size), int(validate_size), filter_params)

    try:
        backend.loop()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        backend.cleanup()
        train_frontend.stop()
        validate_frontend.stop()
        
if __name__ == '__main__':
    main()