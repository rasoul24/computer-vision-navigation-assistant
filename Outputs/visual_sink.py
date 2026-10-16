import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))

from Navigation.decode_image import decode_segmap_cython
from Navigation.bounding_box import get_bounding_boxes
from Navigation.pedestrian_traj import trajectoire
from Core.analyze_frame import predict_model




dict_korean_color = {0: (255, 128, 255), 1: (0, 0, 255), 2: (230, 170, 255), 3: (0, 0, 0), 4: (255, 155, 155), 5: (255, 255, 0), 6: (0, 255, 0)}

palette = {
    0: (128, 64, 128),    # Road (Violet Cityscapes)
    1: (244, 35, 232),    # Sidewalk (Rose Cityscapes) - Surface principale
    2: (110, 110, 110),   # Alley (Gris sombre) - Pour différencier de la route
    3: (200, 0, 0),       # Bike Lane (Rouge Foncé) - Zone de danger
    4: (0, 255, 255),     # Braille Blocks (CYAN ELECTRIQUE) - Doit péter sur le rose !
    5: (255, 165, 0),     # Caution Zone (Orange Vif) - Attention danger
    6: (70, 70, 70),      # Building (Gris foncé Cityscapes)
    7: (190, 153, 153),   # Fence (Beige Cityscapes)
    8: (220, 20, 60),     # Pedestrian (Rouge Cityscapes)
    9: (153, 153, 153),   # Pole (Gris clair Cityscapes)
    10: (107, 142, 35),   # Vegetation (Vert Cityscapes)
    11: (0, 0, 142),      # Vehicle (Bleu foncé Cityscapes)
    12: (102, 102, 156),  # Wall (Bleu gris Cityscapes)
    13: (220, 220, 0),     # Traffic Sign (Jaune Cityscapes)
}


id2label = {0:"Road",1:"SideWalk",2:"Alley",3:"Bike lane", 4:"Braille Blocks", 5:"Caution Zone",6:"Building",7:"Fence", 8:"Pedestrian", 9:"Pole",10:"Vegetation", 11:"Vehicle", 12:"Wall" ,13:"Traffic Sign"}

palette_np = np.zeros((14, 3), dtype=np.uint8)
for class_id, color in palette.items():
    palette_np[class_id] = color


TARGET_CLASSES = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13], dtype=np.int32)

CLASSES_AFFICHE = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13], dtype=np.int32)
#CLASSES_AFFICHE = np.array([8,9,11,12], dtype=np.int32)


def VisualSink(img):


    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)


    prediction_model = predict_model(img=img)
    prediction_short = prediction_model.astype(np.int16)

    img_rgb = decode_segmap_cython(prediction_short, palette_np)

    traj_mask = trajectoire(img_rgb, 5)

    boxes = get_bounding_boxes(prediction_short, TARGET_CLASSES)

    for obj in boxes:

        if obj["id"] in CLASSES_AFFICHE:
            x1, y1, x2, y2 = obj["bbox"]
            label_name = id2label.get(obj["id"], f"ID:{obj['id']}")

            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(img_rgb, label_name, (x1, y1 - 3),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)


    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.title("Segmentation + Bounding Boxes", fontsize=12, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()




