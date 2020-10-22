import numpy as np
import skimage
from PIL import Image, ImageDraw, ImageFont
from mrcnn.model import mold_image
from numpy import expand_dims

import constant as const
import global_variable as gv
from constant import colors


class KangarooDetector:
    """
        Class to extract Entities from Image or PDF.
    """

    def __init__(self, file_path):
        img = Image.open(file_path)
        img = np.array(img)
        img = skimage.color.gray2rgb(img)
        self.img = Image.fromarray(img.astype('uint8'), 'RGB')

    def get_prediction(self):

        image = np.array(self.img)

        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, gv.cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)

        with gv.graph.as_default():
            yhat = gv.model.detect(sample, verbose=0)[0]

        print('# ' + '-' * 40 + '  Result  ' + '-' * 40 + ' #')
        print("Positions: ", yhat['rois'])
        print("Classes: ", yhat['class_ids'])
        print("Scores: ", yhat['scores'])
        print('# ' + '-' * 90 + ' #')

        return yhat

    def draw_bounding_boxes(self, yhat):
        for cls, box, score in zip(yhat['class_ids'], yhat['rois'], yhat['scores']):
            classes = ['background'] + const.classes
            predicted_class = classes[int(cls)]
            label = '{} {:.2f}'.format(predicted_class, score)

            y1, x1, y2, x2 = box
            color = colors[cls]

            draw = ImageDraw.Draw(self.img)
            size = np.floor(1e-2 * self.img.size[1] + 0.5).astype('int32')
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=size)
            label_size = draw.textsize(label, font)

            if y1 - label_size[1] >= 0:
                text_origin = np.array([x1, y1 - label_size[1]])
            else:
                text_origin = np.array([x1, y1 + 1])

            # create Box, Add Text
            draw.rectangle([(x1, y2), (x2, y1)], outline=color, width=3)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
            draw.text(text_origin, label, fill=(255, 255, 255), font=font)
            del draw

        return self.img
