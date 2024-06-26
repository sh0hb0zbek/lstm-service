import torch
import cv2
import easyocr
from matplotlib import pyplot

def load_model(repo_or_dir, model, **kwargs):
    return torch.hub.load(repo_or_dir, model, **kwargs)

def predict(model, img_path):
    image = cv2.imread(img_path)
    predicted_image = model(img_path)
    xyxy = predicted_image.xyxy
    n_detections = xyxy[0]
    bboxes = []
    for i in range(len(n_detections)):
        x1, y1, x2, y2, _, _ = xyxy[0][i].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        detected_object = image[y1:y2, x1:x2]
        bboxes.append([detected_object, [x1, y1, x2, y2]])
    return bboxes

def load_image(img_path):
    return cv2.imread(img_path)

def save_cropped(bboxes, filename='saved.png'):
    file = filename[:-4]
    file_type = filename[-4:]
    for i in range(len(bboxes)):
        box = bboxes[i]
        cv2.imwrite(f'{file}_{i:02d}{file_type}', box)

def plate_translation(bboxes):
    reader = easyocr.Reader(['en'])
    results = []
    for box in bboxes:
        readings = reader.readtext(box)
        plate = ''
        for reading in readings:
            plate += reading[-2]
        results.append(plate)
    return results

def redraw_image(image, translations, positions):
    new_image = image.copy()
    for plate, position in zip(translations, positions):
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(new_image, text=plate, org=(position[0], position[1]-10), fontFace=font,
                          fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(new_image, (position[0], position[1]), (position[2], position[3]), (0, 255, 0), 3)
    return new_image
