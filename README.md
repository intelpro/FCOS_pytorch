# FCOS pytorch implementation based on detectron2 
This is repository of my Re-implementation code of FCOS: Fully Convolutional One-Stage Object Detection - ICCV, 2019, pp. 9627-9636
### Requirement 
	Ubuntu 16.04
	Pytorch >= 1.3
	torchvision
	pycocotools $  pip install cython; pip install -U'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI’
### Build detectron2 from source 
	# install it from a local clone:
	$ git clone https://github.com/facebookresearch/detectron2.git
	$ cd detectron2 && python -m pip install -e .
### Training with coco dataset(4 GPUs)
	python train_net.py --num-gpus 4 --config-file configs/R_50_1x.yaml
### Training with coco dataset(1 GPUs)
	python train_net.py --config-file configs/R_50_1x.yaml \
	--num-gpus 1 SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025
### Run demo with trained model 
	python train_net.py --config-file configs/R_50_1x.yaml \
	--eval-only MODEL.WEIGHTS /path/to/checkpoint_file
## Code reference 
	https://github.com/aim-uofa/AdelaiDet
