from launch import LaunchDescription
from launch.actions import RegisterEventHandler, DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.event_handlers import OnExecutionComplete
from launch.substitutions import LaunchConfiguration

def create_verifier_node(context, params):
    size = context.perform_substitution(params["size"])
    
    size_override = {
        "size": int(int(size)/2),
    }
    
    verifier = Node(
        package="motion_synthesizer",
        executable="verify",
        name="verifier",
        output="screen",
        emulate_tty=True,
        parameters=[
            params,
            size_override
        ]
    )
    
    return [verifier]

def generate_launch_description():
    global_params = {
        "size": LaunchConfiguration("size"),
        "mode": LaunchConfiguration("mode")
    }
     
    detector = Node(
        package="feature_detector",
        executable="node",
        name="detector",
        output="screen",
        emulate_tty=True,
        parameters=[
            global_params
        ]
    )
 
    synthesizer = Node(
        package="motion_synthesizer",
        executable="node",
        name="synthesizer",
        output="screen",
        emulate_tty=True,
        parameters=[
            global_params
        ]
    )
    
    create_verifier = OpaqueFunction(function=create_verifier_node, args=[global_params])

    return LaunchDescription([
        DeclareLaunchArgument('mode'),
        DeclareLaunchArgument('size'),
        RegisterEventHandler(
            event_handler=OnExecutionComplete(
                target_action=synthesizer,
                on_completion=[create_verifier],
            )
        ),
        RegisterEventHandler(
            event_handler=OnExecutionComplete(
                target_action=detector,
                on_completion=[synthesizer],
            )
        ),
        detector
    ])