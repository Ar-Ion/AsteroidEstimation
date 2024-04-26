import feature_descriptor.node
import coffee_nn.train_node
import motion_synthesizer.node
import motion_synthesizer.verify
import simulation.verify_dataset

# This run.py executable is used to allow direct debugging of the ROS2 nodes from VSCode
# If debugging is not needed, please use rosrun or create launch files
if __name__ == "__main__":
    coffee_nn.train_node.main()