from .backend import Backend

# Classical backends
from .akaze import AKAZEBackend
from .brisk import BRISKBackend
from .fast import FASTBackend
from .kaze import KAZEBackend
from .orb import ORBBackend
from .sift import SIFTBackend
from .surf import SURFBackend

# COFFEE backends
from .coffee import UntrainedCOFFEEBackend
from .coffee import TrainedCOFFEEBackend

# ML backends
from .superpoint import SuperpointBackend
from .contextdesc import ContextDescBackend # not working
from .d2net import D2NetBackend
from .delf import DELFBackend # not working
from .disk import DiskBackend
from .geodesc import GeodescBackend # not working
from .keynet import KeyNetBackend # not working
from .lfnet import LFNetBackend
from .r2d2 import R2D2Backend