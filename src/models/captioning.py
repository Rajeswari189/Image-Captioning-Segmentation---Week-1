
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class CaptioningModel:
    """
    Image Captioning using Salesforce BLIP (pretrained).
    Works on CPU and GPU.
    """
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(self, image: Image.Image, max_length: int = 40) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_length=max_length)
        text = self.processor.decode(out_ids[0], skip_special_tokens=True)
        return text.strip()
