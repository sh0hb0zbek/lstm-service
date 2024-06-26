import os

def main():
    # download datasets for Vehicle registration plate
    # download train dataset
    os.system("python downloader.py --classes 'Vehicle registration plate' --type_csv train")
    # download test dataset
    os.system("python downloader.py --classes 'Vehicle registration plate' --type_csv test")
    # download validation dataset
    os.system("python downloader.py --classes 'Vehicle registration plate' --type_csv val")
    # # OR download all dataset at once
    #os.system("python downloader.py --classes 'Vehicle registration plate' --type_csv all")
    
    
    # train model
    os.system('python model.py train --weights yolov5l --download --save-as number_plate_detector.pt --epochs 50 --batch-size 64 --img 416')
    
    # detect registration plate on video
    os.system('python model.py detect --source cars.mp4 --weights ./number_plate_detector.pt')
    
    # detected video (or images) can be found in ~/current/working/directory/runs/detect/exp* folder
    
    from source import *
    from matplotlib import pyplot
    import cv2
    # detect on image by loading trained model
    model = load_model('./', 'custom', path='number_plate_detector.pt', force_reload=True, source='local')
    
    # load image
    image_path = './test_image.jpg
    image = load_image(image)
    
    # detect object on the image
    detection = predict(model, image_path)
    
    
    detected_obj, positions = [], []
    for d, p in detection:
        detected_obj.append(d)
        position.append(p)
    
    # read plate from detected images
    results = plate_translation(detected_obj)
    
    # re-draw image with boundaries
    new_image = redraw_image(image, results, position)
    
    # show the final image
    pyplot.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)))

if __name__ == '__main__':
    main()
