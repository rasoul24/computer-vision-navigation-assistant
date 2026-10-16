import sys
from pathlib import Path
import cv2

from Outputs.visual_sink import VisualSink
from Outputs.text_sink import TextSink
from Outputs.video_sink import VideoSink

def main(mode, img=None, video=None):
    if mode == "vision":
        VisualSink(img)
    elif mode == 'text':
        TextSink(img)
    elif mode == "video":
        VideoSink(video)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Erreur : Arguments manquants.")
        print("Exemple : python main.py vision image_test1")
        print("Exemple : python main.py video tt")
        sys.exit(1)

    mode_arg = sys.argv[1]
    file_name = sys.argv[2]

    img = None
    video = None


    if mode_arg in ["vision", "text"]:
        path_img = Path("Data") / f"{file_name}.jpg"
        img = cv2.imread(str(path_img))

        if img is None:
            print(f"Erreur : Impossible de charger l'image à l'emplacement : {path_img.resolve()}")
            sys.exit(1)


    elif mode_arg == "video":

        video_path = Path("Data") / f"{file_name}.mov"
        video = cv2.VideoCapture(str(video_path))

        if not video.isOpened():
            print(f"Erreur : Impossible d'ouvrir la vidéo à l'emplacement : {video_path.resolve()}")
            sys.exit(1)

    else:
        print(f"Erreur : Mode '{mode_arg}' inconnu. Utilisez 'vision', 'text' ou 'video'.")
        sys.exit(1)

    # Lancement de la fonction principale
    main(mode=mode_arg, img=img, video=video)