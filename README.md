# Mask RCNN

Mask RCNN Model is used for Object Detection

#####Folder Structure:
```sh
Mask RCNN
    ├─── dateset
    │   └─── annots/
    |   └─── images/
    |   └─── classes.txt
    ├─── font/
    ├─── Logger
    │   └─── logger.py
    ├─── logs/
    ├─── pretrained_model/
    ├─── saved_model/
    ├─── config.py
    ├─── constant.py
    ├─── dataset.py
    ├─── global_variable.py
    ├─── kangaroo_detector.py
    ├─── train.py
    ├─── test.py
    ├─── main.py : main flask API
    └───  requirements.txt
```


### Installation

 requires [python](https://www.python.org/download/releases/3.0/) v3(3.6.9) to run.


Installing required packages
```sh
$ pip install -r requirement.txt
```


Installing Mask RCNN
```sh
$ git clone https://github.com/matterport/Mask_RCNN.git
$ cd Mask_RCNN
$ python setup.py install
$ cd ..
$ rm -rf Mask_RCNN
```

### Train Model

Dataset: [Kangaroo Dataset](https://github.com/experiencor/kangaroo)

Pretrained Model: [pretrained model: Coco Dataset](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) 

Copy these data and model and paste it in respective directory

For Training Mask RCNN Model,
```sh
$ python train.py
```
after training, chage saved model path in [constant.py](./constant.py)

### Test Model

For Training Mask RCNN Model,
```sh
$ python test.py
```


### Run Application


For running Flask API,
```sh
$ python main.py
```
