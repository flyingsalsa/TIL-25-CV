from typing import List, Dict, Any
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import supervision as sv
# cv2 is not strictly needed if you stick to PIL for initial image loading and np conversion
# but it's good to have for general CV tasks. You could also do:
# import cv2 # If you prefer cv2 for image loading/manipulation

class CVManager:
    """Manages the CV model with image slicing capabilities."""

    def __init__(self, model_path: str = 'models/best.pt'):
        """Initialize CV Manager.

        This is where you can initialize your model and any static configurations.
        Args:
            model_path (str): Path to the YOLO model file.
        """
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

        # Slicer configuration (can be made parameters of __init__ or cv method)
        self.slice_wh = (640, 640) # Standard YOLOv8 input size, adjust if needed
        self.overlap_ratio_wh = (0.2, 0.2) # Common overlap
        self.iou_threshold = 0.45 # For NMS within slicer post-processing
        self.confidence_threshold = 0.25 # For NMS within slicer post-processing
        
        # Note: supervision's InferenceSlicer uses its own NMS after aggregating
        # slice predictions. The default overlap_filter_strategy is NMS.
        # You can also control the NMS parameters (iou_threshold, confidence_threshold)
        # directly in the InferenceSlicer if using a version that supports it, or
        # apply NMS manually to the aggregated sv.Detections if needed.
        # For newer supervision versions, these are set in the Slicer itself.

    def _slicer_callback(self, image_slice: np.ndarray) -> sv.Detections:
        """
        Callback function for the InferenceSlicer.
        Performs inference on a single image slice and returns sv.Detections.
        """
        if not self.model:
            return sv.Detections.empty()

        # Perform inference on the slice
        # The image_slice is already a NumPy array, which YOLO can take directly.
        results = self.model(image_slice, verbose=False) # verbose=False to suppress YOLO logs

        # Ensure results is not empty and contains detections
        if results and len(results) > 0:
            # Convert Ultralytics YOLO results to_sv.Detections object
            # results[0] corresponds to the detections for the single image slice
            return sv.Detections.from_ultralytics(results[0])
        return sv.Detections.empty()

    def cv(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Performs object detection on an image using slicing.

        Args:
            image_bytes: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions.
            Format: [{"bbox": [x, y, w, h], "category_id": int}, ...]
        """
        if not self.model:
            print("CV Model not initialized. Returning empty predictions.")
            return []

        predictions_for_image: List[Dict[str, Any]] = []
        try:
            # 1. Load image bytes into a PIL Image object, then convert to NumPy array (RGB)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(pil_image) # Now image_np is HxWx3 NumPy array

            # 2. Initialize the InferenceSlicer
            # For newer versions of supervision (e.g., 0.17.0+), NMS parameters are part of Slicer
            slicer = sv.InferenceSlicer(
                callback=self._slicer_callback,
                slice_wh=self.slice_wh,
                overlap_ratio_wh=self.overlap_ratio_wh,
                iou_threshold=self.iou_threshold, # NMS IoU threshold for merging
                confidence_threshold=self.confidence_threshold # NMS confidence for merging
                # overlap_filter_strategy=sv.OverlapFilter.NMS # Default, can be explicit
            )
            
            # For older versions, you might need to apply NMS separately or it uses defaults.
            # Check your supervision version documentation if issues arise.

            # 3. Perform sliced inference
            # The slicer will call `_slicer_callback` for each slice
            # and then aggregate results (e.g., using NMS).
            detections_sv = slicer(image_np)

            # 4. Convert sv.Detections to the required output format
            # detections_sv.xyxy are [xmin, ymin, xmax, ymax]
            # detections_sv.class_id is the class ID
            # detections_sv.confidence is the confidence score (if you need it)

            for i in range(len(detections_sv)):
                xyxy = detections_sv.xyxy[i]
                class_id = detections_sv.class_id[i]
                # confidence = detections_sv.confidence[i] # If you want to include confidence

                x_min, y_min, x_max, y_max = xyxy

                # Convert [xmin, ymin, xmax, ymax] to [x_top_left, y_top_left, width, height]
                x = x_min
                y = y_min
                w = x_max - x_min
                h = y_max - y_min

                # Ensure bbox coordinates are integers
                formatted_bbox = [int(x), int(y), int(w), int(h)]

                predictions_for_image.append({
                    "bbox": formatted_bbox,
                    "category_id": int(class_id)
                    # "confidence": float(confidence) # Optionally add confidence
                })

        except Exception as e:
            print(f"Error during CV processing: {e}")
            # Depending on desired behavior, you might want to re-raise or return empty
            return []
            
        return predictions_for_image
