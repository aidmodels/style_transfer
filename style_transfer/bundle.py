from pkg_resources import resource_filename
from cvpm.bundle import Bundle

class StyleTransferBundle(Bundle):
    PRETRAINED_TOML = resource_filename(__name__, "../pretrained/pretrained.toml")
    CANDY_STYLE_LOCATION = resource_filename(__name__, "../pretrained/candy.pth")
    MOSAIC_STYLE_LOCATION = resource_filename(__name__, "../pretrained/mosaic.pth")
    RAIN_PRINCESS_STYLE_LOCATION = resource_filename(__name__, "../pretrained/rain_princess.pth")
    UDNIE_STYLE_LOCATION = resource_filename(__name__, "../pretrained/udnie.pth")
    ENABLE_TRAIN = False
    SOLVERS = []