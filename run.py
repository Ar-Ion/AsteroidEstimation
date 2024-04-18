import multiprocessing as mp
import feature_detector.node
import feature_descriptor.node
import synthetic_motion.node
import os
import signal

# This run.py executable is used to allow direct debugging of the ROS2 nodes from VSCode
# If debugging is not needed, please use rosrun or create launch files
if __name__ == "__main__":
    feature_descriptor.node.main()
    #ctx = mp.get_context('spawn')
        
    #feature_detector_node = ctx.Process(target=feature_detector.node.main)
    #feature_descriptor_node = ctx.Process(target=feature_descriptor.node.main)
    #synthetic_motion_node = ctx.Process(target=synthetic_motion.node.main)

    #feature_detector_node.start()
    #feature_descriptor_node.start()
    #synthetic_motion_node.start()
    
    #feature_detector_node.join()
    #feature_descriptor_node.join()
    #try:
    #    synthetic_motion_node.join()
    #except KeyboardInterrupt:
    #    pass
    #finally:
    #    synthetic_motion_node.join()
