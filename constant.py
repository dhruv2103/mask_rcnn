EPOCHS = 15

CONFIG_NAME = "kangaroo_cfg"
STEPS_PER_EPOCH = 55

PRETRAINED_MODEL = 'pretrained_model/mask_rcnn_coco.h5'
SAVED_MODEL = 'saved_models'
TRAINED_MODEL = 'saved_models/kangaroo_cfg20200920T1445/mask_rcnn_kangaroo_cfg_0005.h5'

dataset_path = 'dataset'
classes_path = 'dataset/classes.txt'
classes = open(classes_path).read().splitlines()

colors = [(244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139)]