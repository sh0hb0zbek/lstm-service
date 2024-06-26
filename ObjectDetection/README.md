# ObjectDetection_YOLOv5

Model Training, Validating, Detecting (and Ecporting) can be done by executing `model.py`

[YOLO_v5](https://github.com/ultralytics/yolov5.git) repository will be automatically cloned if it does not exist in root directory
```bash
$ python <task | train, detect, val (or export)> ...
```

## Training YOLOv5 model
Train model by using custom datasets

```bash
$ python model.py train --weights <path/to/pre-trained/model.pt>
                        --download (optional)
                        --resume (optional)
                        --exist-ok (optional)
                        --device <e.g. 0,1> (optional)
                        --data <path/to/dataset.yaml | default=ROOT/data/dataset.yaml>
                        --batch-size <int | default=16>
                        --epochs <int | default=100>
                        --optimizer <SGD or Adam or AdamW | default=SGD>
                        --img <int | default=640>
```
- `--download (optional)` - if `--weights` does not exist, use to download [YOLO_v5](https://github.com/ultralytics/yolov5/releases) pre-trained models can be used
- `--resume (optional)` - if `--weights` already a trained model, use to resume training that model
- `--exist-ok (optional)` - store results to the latest trained model's `/exp` directory
- `--device <e.g. 0,1> (optional)` - use for multi-GPU model training

e.g.
```bash
$ python model.py train --weights yolov5s --download --batsh-size 128 --epochs=200 --devices 0,1 
```

## Evaluating model

Evaluate trained model by using test images

```bash
$ python model.py val --weights <path/to/trained/model.pt>
                      --data <path/to/test/images/directory/>
                      --batch-size <int | default=16>
                      --img <int | default=640>
                      --device <e.g. 0,1> (optional)
                      --exist-ok (optional)

```
- `--device <e.g. 0,1> (optional)` - use for multi-GPU model evaluation
- `--exist-ok (optional)` - store results to the latest evaluation's `/exp` directory

## Detecting Objects using trained model

Object detection by using trained model

```bash
$ python model.py detect --weights <path/to/trained/model.pt>
                         --source <path/to/target/data or URL or 0 for webcam>
                         --data <path/to/dataset.yaml | default=REPO_DIR/data/images>
                         --img <int | default=640>
                         --conf-thres <float | default=0.25>
                         --hide-labels (optional)
                         --hide-conf (optinal)
                         --exist-ok (optional)
                         --device <e.g. 0,1> (optional)
```
- `--exist-ok (optional)` - store results to the latest detection's `/exp` directory
- ```bash
  --source 0            # webcam
           img.jpg      # image
           vid.mp4      # video
           path/        # directory
           path/*.jpg   # glob
           'https://youtu.be/...'           # YouTube
           'rtsp://example.com/media.mp4'   # RTSP, RTMP, HTTP stream
  ```

## Translate vehicle registration plate

...

## Downloading dataset images
Code was implemented and edited from [OIDv4_ToolKit](https://github.com/theAIGuysCode/OIDv4_ToolKit.git) repository

```bash
$ python downloader.py --classes <name of classes>
                       --type_csv <type of data>
                       --limit <int | None>
                       --data_dir <path/to/download/data/ | default=ROOT/data/> (optional)
                       --do_print (optional)
```

e.g. 
```bash
$ python downloader.py --classes 'Vehicle registration plate' --type_csv train --limit 500
```
  - `--classes <name of classes>` - list of classes or `.txt` file that contains name of classes
  - `--type_csv <type of data>` - 'train' or 'validation' or 'test' or 'all'
  - `--limit <int | default=None>` - optimal limit on number of images to download (if ommited, download all existing data-images)
  - `--data_dir </path/to/download/data/ | default=ROOT/data/> (optional)` - use if datasets are supposed to be downloaded to specific directory
  - `--do_print (optional)` - use to print logging messages
