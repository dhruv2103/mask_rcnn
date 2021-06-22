# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset

# class that defines and loads the kangaroo dataset
from constant import classes
from util.xml_parser import extract_boxes


class ImageDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define one class
        for i, line in enumerate(classes, 1):
            self.add_class("dataset", i, line)
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]
            # skip all images after 150 if we are building the train set
            if is_train and (filename in listdir(images_dir)[:175]):
                continue
            # skip all images before 150 if we are building the test/val set
            if not is_train and (filename in listdir(images_dir)[175:]):
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, classes, w, h = extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i, (box, class_name) in enumerate(zip(boxes, classes)):
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(class_name))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# if __name__ == '__main__':
#     # train set
#     train_set = KangarooDataset()
#     train_set.load_dataset('dataset', is_train=True)
#     train_set.prepare()
#     print('Classes: ', train_set.class_names)
#     print('Train: %d' % len(train_set.image_ids))
#
#     # test/val set
#     test_set = KangarooDataset()
#     test_set.load_dataset('dataset', is_train=False)
#     test_set.prepare()
#     print('Test: %d' % len(test_set.image_ids))


# if __name__ == '__main__':
#     # train set
#     train_set = KangarooDataset()
#     train_set.load_dataset('dataset', is_train=True)
#     train_set.prepare()
#     # enumerate all images in the dataset
#     for image_id in train_set.image_ids:
#         # load image info
#         info = train_set.image_info[image_id]
#         # display on the console
#         print(info)


# if __name__ == '__main__':
#     # Show 9 Photos with Mask
#     train_set = KangarooDataset()
#     train_set.load_dataset('dataset', is_train=True)
#     train_set.prepare()
#
#     # load an image
#     from matplotlib import pyplot
#     for i in range(9):
#         # define subplot
#         pyplot.subplot(330 + 1 + i)
#         # plot raw pixel data
#         image = train_set.load_image(i)
#         pyplot.imshow(image)
#         # plot all masks
#         mask, _ = train_set.load_mask(i)
#         for j in range(mask.shape[2]):
#             pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
#     # show the figure
#     pyplot.show()

# if __name__ == '__main__':
#     from mrcnn.visualize import display_instances
#     from mrcnn.utils import extract_bboxes
#     # train set
#     train_set = KangarooDataset()
#     train_set.load_dataset('dataset', is_train=True)
#     train_set.prepare()
#     # define image id
#     image_id = 1
#     # load the image
#     image = train_set.load_image(image_id)
#     # load the masks and the class ids
#     mask, class_ids = train_set.load_mask(image_id)
#     # extract bounding boxes from the masks
#     bbox = extract_bboxes(mask)
#     # display image with masks and bounding boxes
#     display_instances(image, bbox, mask, class_ids, train_set.class_names)
