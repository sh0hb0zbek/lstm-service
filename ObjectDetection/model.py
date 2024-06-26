from git.repo.base import Repo
from config import cfg
import requests
import os
import argparse
import pathlib

ROOT_DIR = os.getcwd()
repo_dir = os.path.join(ROOT_DIR, cfg.MODEL.DIR)

def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',
                        metavar="<command> 'train' or 'val' or 'detect' or 'export'",
                        help="'train', 'val', 'detect', 'export'.")
    parser.add_argument('--weights',
                        type=str,
                        default=os.path.join(repo_dir, 'yolov5s.pt'),
                        help='initial weights path')
    parser.add_argument('--download',
                        required=False,
                        action='store_true',
                        help='Download pre-trained YOLO_v5 model')
    parser.add_argument('--save-as',
                        type=str,
                        default = '',
                        required=False,
                        help='Save trained YOLO_v5 model as')
    parser.add_argument('--data',
                        type=str,
                        default=os.path.join(ROOT_DIR, cfg.DATASET),
                        help='dataset.yaml path')
    parser.add_argument('--epochs',
                        type=int,
                        default=100)
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img_size',
                        type=int,
                        default=640,
                        help='train, val image size (pixels)')
    parser.add_argument('--resume',
                        nargs='?',
                        const=True,
                        help='resume most recent training')
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer',
                        type=str,
                        choices=['SGD', 'Adam', 'AdamW'],
                        default='SGD',
                        help='optimizer')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--include',
                        nargs='+',
                        default=['torchscript', 'onnx'],
                        help='torchscript, onnx, openvino, engine, coreml,  saved_model, pb, tflite, edgetpu, tfjs')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.25,
                        help='confidence threshold')
    parser.add_argument('--source',
                        type=str,
                        default=os.path.join(repo_dir, 'data', 'images'),
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--hide-labels',
                        default=False,
                        action='store_true',
                        help='hide labels')
    parser.add_argument('--hide-conf',
                        default=False,
                        action='store_true',
                        help='hide confidences')
    opt = parser.parse_args()
#     if opt.command == 'detect':
#         opt.imgsz = [opt.imgsz]
    if opt.command == 'export':
        if opt.device == '':
            opt.device = 'cpu'
    return opt


if __name__ == '__main__':
    # clone YOLO_v5 GitHub Repository
    if not os.path.exists(repo_dir):
        Repo.clone_from(cfg.MODEL.REPO, repo_dir)
    
    args = parser_arguments()
    if args.download:
        model_name = args.weights
        model_name +=  '' if model_name.endswith('.pt') else '.pt'
        r = requests.get(cfg.MODEL.URL + model_name, allow_redirects=True)
        model_path = os.path.join(repo_dir, model_name)
        open(model_path, 'wb').write(r.content)
        args.weights = model_path
    else:
        model_name = args.weights.split('/')[-1][:-3]
    
    project_dir = os.path.join(ROOT_DIR, 'runs', args.command)
    
    if args.command == 'train':
        command = f'python {os.path.join(repo_dir, "train.py")} --weights {args.weights} --data {args.data} --batch-size {args.batch_size} '
        command += f'--epochs {args.epochs} --optimizer {args.optimizer} --img {args.imgsz} --project {project_dir}'
        if args.resume:
            command += ' --resume'
    elif args.command == 'val':
        if args.data == os.path.join(ROOT_DIR, cfg.DATASET):
            args.data = os.path.join(ROOT_DIR, cfg.TEST)
        command = f'python {os.path.join(repo_dir, "val.py")} --data {args.data} --weights {args.weights} --img {args.imgsz}'
        command += f' --batch-size {args.batch_size} --project {project_dir}'
    elif args.command == 'detect':
        command = f'python {os.path.join(repo_dir, "detect.py")} --data {args.data} --weights {args.weights} --img {args.imgsz} '
        command += f'--source {args.source} --conf-thres {args.conf_thres} --project {project_dir}'
        command += f' --hide-labels {args.hide_labels}' if args.hide_labels else ''
        command += f' --hide-conf {args.hide_conf}' if args.hide_conf else ''
    elif args.command == 'export':
        command = f'python {os.path.join(repo_dir, "export.py")} --data {args.data} --weights {args.weights} --include {args.include}'
    else:
        exit(1)
    if args.exist_ok and args.command != 'export':
        command += ' --exist-ok'
    if args.device != '':
        command += f' --device {args.device}'
    
    # execute the task
    os.system(command)
    
    # save model as ...
    if args.command == 'train':
        # save model
        if args.save_as == '':
            args.save_as = model_name + '_custom_trained.pt'
        else:
            args.save_as = os.path.join(ROOT_DIR, args.save_as)
        
        trained_model_path = os.path.join(project_dir, max(pathlib.Path(project_dir).glob('*/'), key=os.path.getmtime), 'weights', 'best.pt')
        os.system(f'mv {trained_model_path} {args.save_as}')
