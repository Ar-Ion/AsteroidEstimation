import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.event_handlers import OnExecutionComplete
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    run_params = {
        "size": LaunchConfiguration("size"),
        "mode": LaunchConfiguration("mode")
    }

    dataset_params = os.path.join(get_package_share_directory('pipeline'), 'config', 'pipeline.yaml')

    descriptor = Node(
        package="feature_descriptor",
        executable="node",
        name="gen_descriptor",
        output="screen",
        emulate_tty=True,
        parameters=[
            run_params,
            dataset_params
        ]
    )
 
    synthesizer = Node(
        package="motion_synthesizer",
        executable="node",
        name="gen_synthesizer",
        output="screen",
        emulate_tty=True,
        parameters=[
            run_params,
            dataset_params
        ]
    )
    
    create_verifier = OpaqueFunction(function=create_verifier_node, args=[run_params])

    return LaunchDescription([
        DeclareLaunchArgument("mode"),
        DeclareLaunchArgument("size"),
        RegisterEventHandler(
            OnExecutionComplete(
                target_action=synthesizer,
                on_completion=[create_verifier]
            )
        ),
        RegisterEventHandler(
            OnExecutionComplete(
                target_action=descriptor,
                on_completion=[synthesizer]
            )
        ),
        descriptor
    ])

def create_verifier_node(context, run_params, dataset_params):
    size = context.perform_substitution(run_params["size"])
    
    size_override = {
        "size": int(int(size)/2),
    }
    
    verifier = Node(
        package="motion_synthesizer",
        executable="node",
        name="verify",
        output="screen",
        emulate_tty=True,
        parameters=[
            run_params,
            dataset_params,
            size_override
        ]
    )
    
    return [verifier]