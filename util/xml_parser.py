# example of extracting bounding boxes from an annotation file
from xml.etree import ElementTree


# function to extract bounding boxes from an annotation file
def extract_boxes(filename):
    # load and parse the file
    tree = ElementTree.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    classes = list()
    boxes = list()
    for obj in root.findall('.//object'):
        cls = str(obj.find('name').text)
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = [xmin, ymin, xmax, ymax]
        classes.append(cls)
        boxes.append(coors)
    # extract image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    # print('# ' + '-' * 90 + ' #')
    # print("Total Number of Classes : ", len(classes))
    # print("Classes: ", classes)
    # print('# ' + '-' * 90 + ' #')
    return boxes, classes, width, height


# if __name__ == '__main__':
#     # extract details form annotation file
#     boxes, classes, w, h = extract_boxes('dataset/annots/00001.xml')
#     # summarize extracted details
#     print('#' + '-' * 20 + '#')
#     print(boxes, classes, w, h)
#     print('#' + '-' * 20 + '#')
