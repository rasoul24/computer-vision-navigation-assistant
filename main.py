import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Outputs.visual_sink import VisualSink
from Outputs.text_sink import TextSink
import cv2

def main(mode, img):

    if mode == "vision":
        VisualSink(img)

    if mode == 'text':
        TextSink(img)

if __name__ == "__main__":

    mode_arg = sys.argv[1]
    img_name = sys.argv[2]


    path_img = Path("Data/") / f"{img_name}.jpg"

    img = cv2.imread(str(path_img))

    if img is None:
        print(f"Erreur : Impossible de charger l'image à l'emplacement : {path_img.resolve()}")
        sys.exit(1)

    main(mode=mode_arg, img=img)