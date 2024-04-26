import rclpy
import numpy as np
import torch
from matplotlib import pyplot as plt

from astronet_frontends import AsyncFrontend, DriveClientFrontend

def main(args=None):
    rclpy.init(args=args)

    size = 700

    frontend_wrapped = DriveClientFrontend("/home/arion/AsteroidMotionDataset/test", size)
    frontend = AsyncFrontend(frontend_wrapped, AsyncFrontend.Modes.NO_WAIT)
    frontend.start()

    try:
        count = 0

        while count < size:
            data = frontend.receive(blocking=True)
            (c1, c2, f1, f2) = (data.expected_kps[:, 0:2], data.actual_kps, data.expected_features, data.actual_features)

            true_distance = torch.cdist(c1.float(), c2.float())
            predicted_distance = 
            
            epsilon = 8
            s = torch.zeros_like(true_distance)
            s[true_distance < epsilon] = 1
            
            if len(true_distance) > 0 and len(predicted_distance) > 0 and torch.count_nonzero(s) > 0:
                losses.append(self.loss(true_distance, predicted_distance))
                tp.append((s * predicted_distance).sum()/torch.count_nonzero(s))
                tn.append(((1 - s) * (1 - predicted_distance)).sum()/torch.count_nonzero((1 - s)))
                    
            avg_loss = torch.tensor(losses).mean()
            avg_tp = torch.tensor(tp).mean()
            avg_tn = torch.tensor(tn).mean()


            count += 1
            
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        frontend.stop() 
        
if __name__ == '__main__':
    main()