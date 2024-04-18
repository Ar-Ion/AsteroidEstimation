import multiprocessing as mp
import feature_detector
import feature_descriptor

# This run.py executable is used to allow direct debugging of the ROS2 nodes from VSCode
# If debugging is not needed, please use rosrun or create launch files
if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    feature_detector_node = mp.Process(target=feature_detector.node.main)
    feature_descriptor_node = mp.Process(target=feature_descriptor.node.main)
    
    feature_detector_node.start()
    feature_descriptor_node.start()
    
    feature_detector_node.join()
    feature_descriptor_node.join()
