import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    run_params = {
        "train_size": LaunchConfiguration("train_size"),
        "validate_size": LaunchConfiguration("validate_size")
    }

    dataset_params = os.path.join(get_package_share_directory('pipeline'), 'config', 'pipeline.yaml')
     
    trainer = Node(
        package="coffee_nn",
        executable="main_train_node",
        name="main_trainer",
        output="screen",
        emulate_tty=True,  
        parameters=[
            dataset_params,
            run_params
        ]
    )
 
    return LaunchDescription([
        DeclareLaunchArgument("train_size"),
        DeclareLaunchArgument("validate_size"),
        trainer
    ])