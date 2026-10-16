import torch
import torch.nn.functional as F
from pathlib import Path
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def predict_model(img):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = str(Path(r"./Segmentation_Model/"))
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
    model.eval()


    inputs = processor(images=img, return_tensors="pt", do_rescale=False).to(device)

    with torch.no_grad():

        with torch.cuda.amp.autocast():
            outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

    prediction = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    return prediction