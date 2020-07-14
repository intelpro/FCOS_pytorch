# FCOS pytorch implementation based on detectron2 
This is repository of implementation of FCOS: Fully Convolutional One-Stage Object Detection - ICCV, 2019, pp. 9627-9636
### Requirement 
	Ubuntu 16.04
	Pytorch >= 1.3
	torchvision
	pycocotools $  pip install cython; pip install -U'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI’
### Build detectron2 from source 
	# install it from a local clone:
	$ git clone https://github.com/facebookresearch/detectron2.git
	$ cd detectron2 && python -m pip install -e .
### Training with coco dataset 
	sh run.sh
### Run demo with trained model 
	sh run_demo.sh
