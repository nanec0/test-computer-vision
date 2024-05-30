from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import re

class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = ("It will return a list of detected objects along with their categories (Textile Restoration, "
                   "Electronic Restoration, Art Restoration) and confidence scores. If the confidence score is less "
                   "than 60%, it will raise an alert for human validation.")

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        # Generate caption
        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # use 'cuda' if GPU is available

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

   
class ImageProcessingHelper:
    def classify_object(self, object_name):
            print(object_name)
            print("Object Name")
            # Improved logic for classifying objects into restoration categories
            textile_keywords = [
                "cloth", "fabric", "garment", "textile", "towel", "blanket", 
                "clothing", "shoe", "handbag", "leather", "luggage", "curtain", 
                "taxidermy", "linen", "rug", "pillow", "bedspread", "comforter"
            ]
            electronic_keywords = [
                "laptop", "phone", "camera", "electronic", "TV", "tablet", 
                "smartphone", "computer", "printer", "monitor", "speaker", "console"
            ]
            art_keywords = [
                "painting", "sculpture", "art", "picture", "frame", "print", 
                "drawing", "poster", "vase", "china", "lamp", "photo", "album", "candle"
                "mural", "mosaic", "doll", "collection", "coin", "antique", 
                "clock", "toy", "memorabilia", "tapestry", "flag", "textile art",
                "book", "manuscript", "map", "document", "etching", "statuette"
            ]

            object_name_lower = object_name.lower()
            if any(keyword in object_name_lower for keyword in textile_keywords):
                return "Textile Restoration"
            elif any(keyword in object_name_lower for keyword in electronic_keywords):
                return "Electronic Restoration"
            elif any(keyword in object_name_lower for keyword in art_keywords):
                return "Art Restoration"
            else:
                return "Uncategorized"

    def extract_objects_from_caption(self, caption):
        # Extract possible objects from the caption using regular expressions
        object_keywords = re.findall(r'\b\w+\b', caption)
        return [word for word in object_keywords if self.classify_object(word) != "Uncategorized"]
    
    def process_objects(self, response):
        print("Response")
        print(response)
        objects = self.extract_objects_from_caption(response)
        print("extract_objects_from_caption")
        print(objects)
        detections = ""
        validation_required = False

        for object_name in objects:
            category = self.classify_object(object_name)
            # Assume confidence score as a placeholder since BLIP does not provide it
            confidence = 1.0
            detection_info = f'Object: {object_name}, Category: {category}, Confidence: {confidence * 100:.2f}%\n'
            detections += detection_info

            if confidence < 0.6:
                validation_required = True

        if validation_required:
            detections += "\nAlert: Human validation required due to low confidence in some classifications."

        return detections


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = ("Use this tool when given the path to an image that you would like to detect objects. "
                   "It will return a list of detected objects along with their categories (Textile Restoration, "
                   "Electronic Restoration, Art Restoration) and confidence scores. If the confidence score is less "
                   "than 60%, it will raise an alert for human validation.")

    def _run(self, img_path):
        image = Image.open(img_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        validation_required = False

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            object_name = model.config.id2label[int(label)]
            category = self.classify_object(object_name)
            confidence = float(score)

            detection_info = f'Object: {object_name}, Category: {category}, Confidence: {confidence * 100:.2f}%\n'
            detections += detection_info

            if confidence < 0.6:
                validation_required = True

        if validation_required:
            detections += "\nAlert: Human validation required due to low confidence in some classifications."

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

    def classify_object(self, object_name):
        # Improved logic for classifying objects into restoration categories
        textile_keywords = ["cloth", "fabric", "garment", "textile", "towel", "blanket", "clothing", "shoe", "handbag", "leather", "luggage", "curtain", "taxidermy", "linen"]
        electronic_keywords = ["laptop", "phone", "camera", "electronic", "TV", "tablet", "smartphone", "computer"]
        art_keywords = ["painting", "sculpture", "art", "picture", "frame", "print", "drawing", "poster", "vase", "china", "lamp", "photo", "album", "mural", "mosaic", "doll", "collection", "coin", "antique", "clock", "rug", "toy", "memorabilia", "tapestry", "flag", "textile art"]

        object_name_lower = object_name.lower()
        if any(keyword in object_name_lower for keyword in textile_keywords):
            return "Textile Restoration"
        elif any(keyword in object_name_lower for keyword in electronic_keywords):
            return "Electronic Restoration"
        elif any (keyword in object_name_lower for keyword in art_keywords):
            return "Art Restoration"
        else:
            return "Uncategorized"
