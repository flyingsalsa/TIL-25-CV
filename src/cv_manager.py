
from typing import Any
import torch
from ultralytics import YOLO
import io
from typing import Any, List # Updated List to be List for modern Python type hinting
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class CVManager:
    """Manages the CV model."""
    
    def __init__(self):
        """Initialize CV Manager."""
        try:
            self.model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path='models/best.pt',
                confidence_threshold=0.3,
                device= 'cuda' if torch.cuda.is_available() else 'cpu'
            )
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None 

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """
        if not self.model:
            print("CV Model not initialized. Returning empty predictions.")
            return []

        predictions_for_image: List[dict[str, Any]] = []
        try:
            # 1. Load image bytes into a PIL Image object
            # The input image is already in bytes, so we can use io.BytesIO
            pil_image = Image.open(io.BytesIO(image))
            
            
            result = get_sliced_prediction(
                pil_image,
                self.model,
                slice_height=512,  # Adjust as needed
                slice_width=512,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            
            # Process each prediction
            for obj in result.object_prediction_list:
                bbox = obj.bbox.to_xywh()
                category_id = obj.category.id

                # Ensure bbox coordinates are integers
                formatted_bbox = [int(coord) for coord in bbox]

                predictions_for_image.append({
                    "bbox": formatted_bbox,
                    "category_id": category_id
                })

        except Exception as e:
            print(f"Error during SAHI inference: {e}")

        return predictions_for_image
         
