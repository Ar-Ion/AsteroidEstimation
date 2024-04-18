import rclpy

from astronet_frontends import AsyncFrontend, DriveClientFrontend
from feature_descriptor.trainer import Trainer
import cProfile

def main(args=None):
    rclpy.init(args=args)

    train_size = 7000
    validate_size = 1500

    train_frontend_wrapped = DriveClientFrontend("/home/arion/AsteroidMotionDataset/train", train_size)
    train_frontend = AsyncFrontend(train_frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    train_frontend.start()
    
    validate_frontend_wrapped = DriveClientFrontend("/home/arion/AsteroidMotionDataset/validate", validate_size)
    validate_frontend = AsyncFrontend(validate_frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    validate_frontend.start()
    
    backend = Trainer(train_frontend, validate_frontend, train_size, validate_size)

    try:
        backend.loop()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        train_frontend.stop()
        
if __name__ == '__main__':
    cProfile.run("main()")