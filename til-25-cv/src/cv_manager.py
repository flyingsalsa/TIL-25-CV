"""Manages the CV model."""

from typing import Any


class CVManager:
    """Manages the CV model."""

    def __init__(self):
        """Initialize CV Manager.

        This is where you can initialize your model and any static configurations.
        """
        pass

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """
        # Your inference code goes here.
        return []
