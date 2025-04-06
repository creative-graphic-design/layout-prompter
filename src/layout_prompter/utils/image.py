import base64
import colorsys
import io
from typing import List, Tuple

from PIL import Image

from layout_prompter.typehints import PilImage


def pil_to_base64(image: PilImage) -> str:
    """Convert a Pillow image to a base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(base64_str: str) -> PilImage:
    """Convert a base64 string to a Pillow image."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image


def generate_color_palette(n_colors: int) -> List[Tuple[int, int, int]]:
    """Generate a color palette with n_colors using HSV color space."""

    def get_rgb(ci: int, cn: int):
        hue = ci / cn
        rgb = colorsys.hsv_to_rgb(h=hue, s=1, v=1)
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    palette = [get_rgb(ci=ci, cn=n_colors) for ci in range(n_colors)]
    return palette
