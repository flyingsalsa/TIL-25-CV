
from typing import Any
from ultralytics import YOLO
import io
from typing import Any, List # Updated List to be List for modern Python type hinting
from PIL import Image
from sahi import AutoDetectionModel


class CVManager:
    """Manages the CV model."""
  
        """Initialize CV Manager.

        This is where you can initialize your model and any static configurations.
        """
        try:
            self.model = YOLO('models/best.pt')
            print("YOLO model loaded successfully from models/best.pt")
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

            # 2. Perform inference
            # results is a list of Results objects. For a single image, we take results[0].
            # verbose=False to prevent YOLO from printing to console during inference.
            results = self.model(pil_image, verbose=False)

            if results and len(results) > 0:
                # Access the detections for the first (and only) image
                detections = results[0]

                # Iterate over each detected bounding box
                for box in detections.boxes:
                    # box.xywh provides [x_center, y_center, width, height] as a tensor
                    # We need to convert it to [x_top_left, y_top_left, width, height]
                    xywh_center = box.xywh[0].tolist() # Get the first (and only) box's data as a list
                    
                    xc, yc, w, h = xywh_center
                    
                    # Calculate top-left x and y
                    x = xc - (w / 2)
                    y = yc - (h / 2)
                    
                    # Ensure bbox coordinates are integers as often expected
                    formatted_bbox = [int(x), int(y), int(w), int(h)]
                    
                    # Get the class ID
                    category_id = int(box.cls[0].item()) # .cls is a tensor, .item() gets the Python number

                    predictions_for_image.append({
                        "bbox": formatted_bbox,
                        "category_id": category_id
                    })
            # If no objects are detected, predictions_for_image will remain empty,
            # which is the correct behavior as per the problem statement.

        except FileNotFoundError:
      
            print(f"Error: Could not identify image from bytes.")
        except Exception as e:
            # Catch any other errors during image processing or inference
            print(f"An error occurred during CV processing: {e}")

        return predictions_for_image
