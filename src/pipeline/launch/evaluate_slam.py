import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, DeclareLaunchArgument, OpaqueFunction
from launch.conditions import UnlessCondition
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnExecutionComplete

def generate_launch_description():
    run_params = {
        "size": LaunchConfiguration("size"),
        "mode": "test",
        "descriptor_config.backend": LaunchConfiguration("backend"),
        "eval.path": LaunchConfiguration("output"),
        "skip_synthesis": LaunchConfiguration("skip_synthesis")
    }

    dataset_params = os.path.join(get_package_share_directory('pipeline'), 'config', 'pipeline.yaml')

    create = OpaqueFunction(function=create_nodes, args=[run_params, dataset_params])

    return LaunchDescription([
        DeclareLaunchArgument("skip_synthesis", default_value=["False"]),
        DeclareLaunchArgument("size"),
        DeclareLaunchArgument("output", default_value=["/home/arion/AsteroidEvaluation"]),
        DeclareLaunchArgument("backend"),
        create
    ])

def create_nodes(context, run_params, dataset_params):
    #eval_dir = context.perform_substitution(run_params["eval.path"])
    #backend = context.perform_substitution(run_params["descriptor_config.backend"])
    #size = context.perform_substitution(run_params["size"])
    #skip_synthesis = context.perform_substitution(run_params["skip_synthesis"])
    
    #matcher_params = os.path.join(get_package_share_directory('pipeline'), 'config', backend + '.yaml')
    
    slam = Node(
        package="slam",
        executable="node",
        name="eval_slam",
        output="screen",
        emulate_tty=True,
        parameters=[
            run_params,
            dataset_params
        ]
    )
    
    return [slam]