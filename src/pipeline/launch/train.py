from launch import LaunchDescription
from launch.actions import RegisterEventHandler, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.event_handlers import OnExecutionComplete
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    global_params = {
        "train_size": LaunchConfiguration("train_size"),
        "validate_size": LaunchConfiguration("validate_size"),
        "iter_ratio": LaunchConfiguration("iter_ratio")
    }
     
    descriptor = Node(
        package="coffee_nn",
        executable="train_node",
        name="coffee_nn",
        output="screen",
        emulate_tty=True,  
        parameters=[
            global_params
        ]
    )
 
    return LaunchDescription([
        DeclareLaunchArgument("train_size"),
        DeclareLaunchArgument("validate_size"),
        DeclareLaunchArgument("iter_ratio", default_value="64"),
        descriptor
    ])