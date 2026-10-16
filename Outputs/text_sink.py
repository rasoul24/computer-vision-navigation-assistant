import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Navigation.pedestrian_traj import trajectoire, vecteur_directeur, check_collisions, generer_alertes_collision
from Navigation.decode_image import decode_segmap_cython

from Navigation.bounding_box import (
    get_bounding_boxes,
    generer_description,
    get_objets_position
)
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
CLASSES_AFFICHE = np.array([8,9,11,12], dtype=np.int32)


def TextSink(img):


    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)


    prediction_model = predict_model(img=img)
    prediction_short = prediction_model.astype(np.int16)

    img_rgb = decode_segmap_cython(prediction_short, palette_np)



    traj_mask = trajectoire(img_rgb, 5)
    trajectoire_float = traj_mask.astype(np.float32)
    v_dir = vecteur_directeur(trajectoire_float)


    boxes = get_bounding_boxes(prediction_short, TARGET_CLASSES)

    collisions = check_collisions(trajectoire_float, boxes)
    message_collision = generer_alertes_collision(collisions, id2label)

    positions = get_objets_position(boxes, v_dir, id2label)
    phrases = generer_description(positions)

    print("phrases = ", phrases)
    print("collisions = ", message_collision)




