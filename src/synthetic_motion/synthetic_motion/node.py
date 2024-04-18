import rclpy

from astronet_frontends import AsyncFrontend, DriveClientFrontend, DriveServerFrontend
from .generator import MotionGenerator
 
def main(args=None):
    rclpy.init(args=args)

    input_size = 1500
    output_size = 15000

    client_wrapped = DriveClientFrontend("/home/arion/AsteroidFeatureDataset/test", input_size)
    server_wrapped = DriveServerFrontend("/home/arion/AsteroidMotionDataset/test", output_size)

    client = AsyncFrontend(client_wrapped, AsyncFrontend.Modes.NO_WAIT)
    server = AsyncFrontend(server_wrapped, AsyncFrontend.Modes.WAIT)

    client.start()
    server.start()
    
    backend = MotionGenerator(client, server, input_size, output_size)

    try:
        backend.loop()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
        server.stop()    

if __name__ == '__main__':
    main()