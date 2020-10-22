from mrcnn.config import Config

# define a configuration for the model
from constant import classes, STEPS_PER_EPOCH, CONFIG_NAME


class TrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = CONFIG_NAME
    # Number of classes (background + kangaroo)
    NUM_CLASSES = 1 + len(classes)
    # Number of training steps per epoch
    STEPS_PER_EPOCH = STEPS_PER_EPOCH


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = CONFIG_NAME
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + len(classes)
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
