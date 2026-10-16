import cv2
from pathlib import Path
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os

try:
    from medial_axis import trajectoire, vecteur_directeur, check_collisions, generer_alertes_collision
    from description_image import decode_segmap_cython
    from bounding_box import get_bounding_boxes, generer_description, get_position_objets
    from prediction_model import predict_model
    print("Modules Cython chargés.")
except ImportError as e:
    print(f"Erreur Cython : {e}")

# --- CONFIGURATION ---
MODEL_DIR = str(Path("./Modele_Complet/"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

# Variables de configuration
TARGET_CLASSES = np.array([0, 1, 4, 5, 6, 8, 9, 11, 12, 13], dtype=np.int32)
CLASSES_AFFICHE = np.array([8, 9, 11, 12], dtype=np.int32)

IA_RES = (512, 512)       # Résolution d'entrée pour l'IA
DISPLAY_RES = (720, 480)  # Résolution de chaque volet d'affichage (Largeur, Hauteur)

# Modèle & Palette de couleurs
palette_np = np.zeros((14, 3), dtype=np.uint8)
palette = {
    0:  (128, 64, 128), # Road (Violet)
    1:  (244, 35, 232), # SideWalk (Rose)
    4:  (0, 255, 255),  # Braille Blocks (Cyan)
    5:  (220, 190, 40), # Caution Zone (Orange/Jaune)
    6:  (70, 70, 70),   # Building (Gris)
    8:  (220, 20, 60),  # Pedestrian (Rouge)
    9:  (153, 153, 153),# Pole (Gris clair)
    11: (0, 0, 142),    # Vehicle (Bleu foncé)
    12: (102, 102, 156),# Wall (Mauve/Gris)
    13: (220, 220, 0)   # Traffic Sign (Jaune vif)
}

for cid, col in palette.items(): palette_np[cid] = col
id2label = {0:"Road", 1:"SideWalk", 4:"Braille Blocks", 5:"Caution Zone", 6:"Building", 8:"Pedestrian", 9:"Pole", 11:"Vehicle", 12:"Wall", 13:"Traffic Sign"}

# Chargement du modèle et du processeur
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR, local_files_only=True).to(device)

video_path = "video_demo.mov"
video = cv2.VideoCapture(video_path)

def process_video(video, model, processor):
    if not video.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Récupération des informations de la vidéo source pour l'export
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25.0  # Valeur par défaut si non détectée

    # Création du dossier de sortie s'il n'existe pas
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(output_dir / "resultat_demo.mp4")

    # Configuration du VideoWriter (Codec 'mp4v' / Largeur combinée = DISPLAY_RES[0] * 2, Hauteur = DISPLAY_RES[1])
    combined_width = DISPLAY_RES[0] * 2
    combined_height = DISPLAY_RES[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (combined_width, combined_height))

    kernel = np.ones((5, 5), np.uint8)
    print("Début du traitement et de l'enregistrement de la vidéo...")

    frame_count = 0
    while True:
        ret, frame_bgr = video.read()
        if not ret:
            break

        # 1. Image originale redimensionnée pour le volet de GAUCHE
        frame_cam = cv2.resize(frame_bgr, DISPLAY_RES)

        # 2. Préparation pour l'IA (RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, IA_RES)

        # 3. Prédiction
        prediction = predict_model(img=frame_small)

        # 4. Post-traitement du masque
        mask = cv2.resize(prediction.astype(np.int16), DISPLAY_RES, interpolation=cv2.INTER_NEAREST)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 5. Décodage et tracé des bounding boxes pour le volet de DROITE
        img_rgb = decode_segmap_cython(mask.astype(np.int16), palette_np)
        seg_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        try:
            boxes = get_bounding_boxes(mask, TARGET_CLASSES)
            for obj in boxes:
                if obj["id"] in CLASSES_AFFICHE:
                    x1, y1, x2, y2 = obj["bbox"]
                    label_name = id2label.get(obj["id"], "Object")
                    cv2.rectangle(seg_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(seg_bgr, label_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            pass

        skel = trajectoire(img_rgb, 5).astype(np.float32)
        seg_bgr[skel > 0] = (0, 255, 255)

        v_dir = vecteur_directeur(skel.astype(np.float32))
        center_x, center_y = DISPLAY_RES[0] // 2, DISPLAY_RES[1] - 50
        if not np.isnan(v_dir[0]):
            end_point = (int(center_x + v_dir[0] * 50), int(center_y + v_dir[1] * 50))
            cv2.arrowedLine(seg_bgr, (center_x, center_y), end_point, (255, 255, 255), 3)

        # 6. Fusion côte à côte (GAUCHE = Caméra | DROITE = Modèle)
        combined_view = np.hstack((frame_cam, seg_bgr))

        # 7. Écriture de la frame dans le fichier vidéo de sortie
        out.write(combined_view)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"{frame_count} frames traitées...")

    # Libération des ressources
    video.release()
    out.release()
    print(f"Traitement terminé ! Vidéo sauvegardée avec succès dans : {output_file}")

# Lancement
process_video(video, model, processor)