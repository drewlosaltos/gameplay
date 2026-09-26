"""Serializable image geometry shared by dataset and full-video evaluation."""
from PIL import Image, ImageOps


class LetterboxResize:
    """Fit the entire RGB image without distortion, with centered gray padding."""

    def __init__(self, image_size):
        self.height, self.width = map(int, image_size)
        if min(self.height, self.width) <= 0:
            raise ValueError('Image dimensions must be positive')

    def __call__(self, image):
        scale = min(self.width / image.width, self.height / image.height)
        width = max(1, min(self.width, round(image.width * scale)))
        height = max(1, min(self.height, round(image.height * scale)))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        left, top = (self.width-width)//2, (self.height-height)//2
        return ImageOps.expand(image, (left, top, self.width-width-left,
                                      self.height-height-top), fill=(128,128,128))
