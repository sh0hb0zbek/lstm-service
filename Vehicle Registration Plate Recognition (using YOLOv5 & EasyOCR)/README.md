### Copy the source.py
### For detection of vehicle licence plate, use the following trained model weights
https://drive.google.com/file/d/1BDlyY83kvk4Cf2RUtatvsMLKpRCvWa2U/view?usp=sharing
### Download model weights (license_plate_detection.pt)
```bash
!gdown 1BDlyY83kvk4Cf2RUtatvsMLKpRCvWa2U
```

### Training
```bash
python train.py --batch 64 --data dataset.yaml --weights yolov5l.pt --epoch 10
```
Multiple GPUs
```bash
python train.py --batch 128 --data dataset.yaml --weights yolov5l.pt --device 0,1,2,3 --epoch 10
```

### Model was trained by using YOLOv5 pre-trained model weights. It can be loaded in PyTorch.
### Model detects the license plate and gets bounding boxes.
### EasyOCR was used for translating the plate number.
config file 
  model weight links, model weight type
  ...
  model weight link, model weight type
  dataset link /type
  Yolo download git link
  

1.  make inew main image text detection
    args: step type: train/test, dataset link, data set type, test image_path, Evaluation results path, evaluation true/flase, test_predicted_ image_path, model weight link, model weight type,  model weight, name of last trained model, name of best trained model, batch size, yaml fil ename, epoch value, device list
    load dataset (main type) ---split to train, test, evaluation (percentage aruments)
  if (steptype==train)
   { !gdown 1BDlyY83kvk4Cf2RUtatvsMLKpRCvWa2U
     load dataset (train type)
    python train.py --batch 128 --data dataset.yaml --weights yolov5l.pt --device 0,1,2,3 --epoch 10
    or
    python train.py --batch 64 --data dataset.yaml --weights yolov5l.pt --epoch 10
   }
   load dataset (test type)
   
   !git clone https://github.com/ultralytics/yolov5.git
   load model
   image_path = './test3.jpg'
   detection = predict(model, image_path)
   image = load_image(image_path)
   detected_img = []
   position = []
   for d, p in detection:
       detected_img.append(d)
        position.append(p)
   results = plate_translation(detected_img)
   ew_image = redraw_image(image, results, position)
   pyplot.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
   print results
   
    if (evaluation==true)
   {
   load dataset (evalaution type)
   Evaluation function (arguments) 
   Evaluation results
   }
