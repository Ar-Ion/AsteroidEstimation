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
        "backend.type": LaunchConfiguration("backend"),
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
    eval_dir = context.perform_substitution(run_params["eval.path"])
    backend = context.perform_substitution(run_params["backend.type"])
    size = context.perform_substitution(run_params["size"])
    skip_synthesis = context.perform_substitution(run_params["skip_synthesis"])
    
    benchmarker_params = os.path.join(get_package_share_directory('pipeline'), 'config', backend + '.yaml')

    descriptor_io_override = {
        "output.path": eval_dir + "/" + backend + "/Features"
    }

    synthesizer_io_override = {
        "input.path": eval_dir + "/" + backend + "/Features",
        "output.path": eval_dir + "/" + backend + "/Motion"
    }

    benchmarker_io_override = {
        "input.path": eval_dir + "/" + backend + "/Motion",
        "output.path": eval_dir + "/" + backend + "/Benchmark"
    }
    
    size_override = {
        "size": int(int(size)/2),
    }
    
    descriptor = Node(
        package="feature_descriptor",
        executable="node",
        name="eval_descriptor",
        output="screen",
        emulate_tty=True,
        parameters=[
            run_params,
            dataset_params,
            descriptor_io_override
        ]
    )

    synthesizer = Node(
        package="motion_synthesizer",
        executable="node",
        name="eval_synthesizer",
        output="screen",
        emulate_tty=True,
        parameters=[
            run_params,
            dataset_params,
            synthesizer_io_override
        ]
    )

    benchmarker = Node(
        package="benchmark",
        executable="node",
        name="eval_benchmarker",
        output="screen",
        emulate_tty=True,
        parameters=[
            run_params,
            dataset_params,
            benchmarker_params,
            benchmarker_io_override,
            size_override
        ]
    )
    
    if skip_synthesis != "False":
        return [benchmarker]
    else:
        return [
            RegisterEventHandler(
                OnExecutionComplete(
                    target_action=synthesizer,
                    on_completion=[benchmarker]
                )
            ),
            RegisterEventHandler(
                OnExecutionComplete(
                    target_action=descriptor,
                    on_completion=[synthesizer]
                )
            ),
            descriptor
        ]