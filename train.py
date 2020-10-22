import os

import keras
from mrcnn.model import MaskRCNN
from config import TrainConfig
from constant import dataset_path, EPOCHS, PRETRAINED_MODEL, SAVED_MODEL
from dataset import ImageDataset


def train():
    # prepare train set
    train_set = ImageDataset()
    train_set.load_dataset(dataset_path, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # prepare test/val set
    test_set = ImageDataset()
    test_set.load_dataset(dataset_path, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    # prepare config
    config = TrainConfig()
    config.display()
    # define the model
    os.mkdir(SAVED_MODEL) if not os.path.isdir(SAVED_MODEL) else None
    model = MaskRCNN(mode='training', model_dir=SAVED_MODEL, config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights(PRETRAINED_MODEL, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # train weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=EPOCHS, layers='heads')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()
