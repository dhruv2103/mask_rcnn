from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.model import MaskRCNN
from config import TrainConfig, PredictionConfig
from constant import dataset_path, EPOCHS, PRETRAINED_MODEL, SAVED_MODEL, TRAINED_MODEL
from dataset import ImageDataset

from numpy import expand_dims, mean
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        # define subplot
        pyplot.subplot(n_images, 2, i * 2 + 1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        pyplot.subplot(n_images, 2, i * 2 + 2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
    # show the figure
    pyplot.show()


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP


if __name__ == '__main__':
    # load the train dataset
    train_set = ImageDataset()
    train_set.load_dataset(dataset_path, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # load the test dataset
    test_set = ImageDataset()
    test_set.load_dataset(dataset_path, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir=SAVED_MODEL, config=cfg)
    # load model weights
    model.load_weights(TRAINED_MODEL, by_name=True)

    # evaluate model on training dataset
    train_mAP = evaluate_model(train_set, model, cfg)
    print("Train mAP: %.3f" % train_mAP)
    # evaluate model on test dataset
    test_mAP = evaluate_model(test_set, model, cfg)
    print("Test mAP: %.3f" % test_mAP)

    # # plot predictions for train dataset
    # plot_actual_vs_predicted(train_set, model, cfg)
    # # plot predictions for test dataset
    # plot_actual_vs_predicted(test_set, model, cfg)
