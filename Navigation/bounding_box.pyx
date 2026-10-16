import numpy as np
cimport numpy as np
cimport cython
import cv2
from libc.math cimport sqrt


CLASS_THRESHOLDS = {
    4: 0.1,  # Braille Blocks
    5: 0.1,   # Caution Zone
    8: 0.01,   # Pedestrian
    9: 0.02,  # Pole
    11: 0.1,  # Vehicle
    13: 0.02  # Traffic Sign
}

CLASSES_AFFICHE = np.array([8,9,11,12], dtype=np.int32)
id2label = {4:"Braille Blocks", 5:"Caution Zone", 8:"Pedestrian", 9:"Pole", 11:"Vehicle", 13:"Traffic Sign"}

@cython.boundscheck(False)
@cython.wraparound(False)

def get_bounding_boxes(short[:, :] pred_mask, int[:] target_ids, float default_min_area_ratio=0.1):
    cdef int h = pred_mask.shape[0]
    cdef int w = pred_mask.shape[1]

    cdef float total_area = <float>(h * w)
    cdef int i, r, c, class_id, n_targets = target_ids.shape[0]
    cdef float current_min_area

    results = []

    cdef np.ndarray[np.uint8_t, ndim=2] binary_mask_np = np.zeros((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] binary_mask = binary_mask_np



    for i in range(n_targets):
        class_id = target_ids[i]


        ratio = CLASS_THRESHOLDS.get(class_id, default_min_area_ratio)
        current_min_area = ratio * total_area

        with nogil:
            for r in range(h):
                for c in range(w):
                    if pred_mask[r, c] == class_id:
                        binary_mask[r, c] = 255
                    else:
                        binary_mask[r, c] = 0

        contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= current_min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                results.append({
                    "id": class_id,
                    "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
                    "area": int(area)
                })

    return results



@cython.boundscheck(False)
@cython.wraparound(False)
def get_objets_position(list boxes, float[:] vecteur_directeur, dict id2label):

    cdef int H = 512
    cdef int W = 512
    cdef float x_0 = W / 2.0
    cdef float y_0 = <float>H

    cdef list results = []
    cdef float a = vecteur_directeur[0]
    cdef float b = vecteur_directeur[1]


    for obj in boxes:
        x1, y1, x2, y2 = obj["bbox"]
        cx = (x1 + x2) / 2.0
        cy = y2

        vx = cx - x_0
        vy = cy - y_0
        norme = sqrt(vx*vx + vy*vy)

        if norme < 1e-6: continue

        vx_norm = vx / norme
        vy_norm = vy / norme

        # Produit scalaire & Déterminant
        ps = vx_norm * a + vy_norm * b
        det = a * vy_norm - b * vx_norm

        pos_horiz = "on the left" if det < -0.1 else ("on the right" if det > 0.1 else "aligned")
        pos_vert = "in front"

        label_name = id2label.get(obj["id"], "object")

        results.append({
            "label": label_name,
            "position": f" {pos_vert} {pos_horiz}",
            "bbox": obj["bbox"],
            "area": obj["area"]
        })

    return results







def generer_description(list position_objets):
    """
    Génère une description vocale concise à partir des objets détectés.
    Les objets sont priorisés selon leur nature, puis regroupés
    par catégorie et position.
    """

    if not position_objets:
        return "No relevant objects detected in the current frame."

    priorites = {
        "Pedestrian": 0,
        "Vehicle": 1,
        "Pole": 2,
        "Traffic Sign": 3,
        "Caution Zone": 4,
        "Braille Blocks": 5,
    }

    objets_tries = sorted(
        position_objets,
        key=lambda obj: (
            priorites.get(obj["label"], 99),
            -obj.get("area", 0)
        )
    )

    annonces = []
    deja_annonces = set()

    for obj in objets_tries:
        label = obj["label"]
        position = " ".join(obj["position"].split())

        # Ne pas répéter la même catégorie au même endroit.
        cle = (label, position)
        if cle in deja_annonces:
            continue
        deja_annonces.add(cle)

        if label == "Pedestrian":
            phrase = f"Pedestrian {position}."

        elif label == "Vehicle":
            phrase = f"Vehicle {position}."

        elif label == "Pole":
            phrase = f"Pole {position}."

        elif label == "Traffic Sign":
            phrase = f"Traffic sign {position}."

        elif label == "Caution Zone":
            phrase = f"Tactile warning surface {position}."

        elif label == "Braille Blocks":
            phrase = f"Tactile paving detected {position}."

        else:
            continue

        annonces.append(phrase)

    if not annonces:
        return "No relevant objects detected in the current frame."

    return " ".join(annonces)





@cython.boundscheck(False)
@cython.wraparound(False)
def consignes_navigations(list boxes, list collisions, float[:] vecteur_directeur, dict id2label, int image_width):
    cdef list consignes_nav = []
    cdef int center_x = image_width // 2
    cdef float obj_x_min, obj_x_max, obj_center_x

    # 1. Si aucune collision n'est détectée, on confirme la marche
    if len(collisions) == 0:
        return ["Chemin libre, continuez tout droit"]

    for col_idx in collisions:
        # col_idx est l'index de la box qui pose problème dans la liste 'boxes'
        # On suppose que chaque box est un tuple/list (x_min, y_min, x_max, y_max, class_id)
        box = boxes[col_idx]
        obj_x_min = box[0]
        obj_x_max = box[2]
        obj_id = box["id"]

        nom_objet = id2label.get(obj_id, "Obstacle")
        obj_center_x = (obj_x_min + obj_x_max) / 2

        # Cas particulier pour les piétons (souvent mobiles, on reste prudent)
        if nom_objet == "Pedestrian":
            consignes_nav.append("Attention, piéton devant. Ralentissez.")
            continue

        # Logique d'évitement latérale
        if obj_center_x < center_x:
            # L'objet est plutôt à gauche, on suggère d'aller à droite
            consignes_nav.append(f"{nom_objet} à gauche, déviez légèrement à droite.")
        else:
            # L'objet est plutôt à droite, on suggère d'aller à gauche
            consignes_nav.append(f"{nom_objet} à droite, déviez légèrement à gauche.")

    return consignes_nav