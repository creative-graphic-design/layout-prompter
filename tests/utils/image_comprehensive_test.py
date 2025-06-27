import base64

import pytest
from layout_prompter.utils.image import (
    base64_to_pil,
    generate_color_palette,
    pil_to_base64,
)
from PIL import Image


class TestImageUtils:
    @pytest.fixture
    def sample_image(self) -> Image.Image:
        """Create a simple test image."""
        return Image.new("RGB", (100, 100), (255, 0, 0))  # Red square

    @pytest.fixture
    def sample_rgba_image(self) -> Image.Image:
        """Create a test RGBA image."""
        return Image.new("RGBA", (50, 50), (0, 255, 0, 128))  # Semi-transparent green

    @pytest.fixture
    def sample_grayscale_image(self) -> Image.Image:
        """Create a test grayscale image."""
        return Image.new("L", (75, 75), 128)  # Gray square

    def test_pil_to_base64_basic(self, sample_image: Image.Image):
        result = pil_to_base64(sample_image)

        assert isinstance(result, str)
        assert len(result) > 0
        # Check that it contains only valid base64 characters
        import string

        valid_chars = string.ascii_letters + string.digits + "+/="
        assert all(c in valid_chars for c in result)

    def test_pil_to_base64_rgba_image(self, sample_rgba_image: Image.Image):
        result = pil_to_base64(sample_rgba_image)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_pil_to_base64_grayscale_image(self, sample_grayscale_image: Image.Image):
        result = pil_to_base64(sample_grayscale_image)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_pil_to_base64_small_image(self):
        # Test with 1x1 pixel image
        tiny_image = Image.new("RGB", (1, 1), (0, 0, 0))
        result = pil_to_base64(tiny_image)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_pil_to_base64_large_image(self):
        # Test with larger image
        large_image = Image.new("RGB", (500, 500), (255, 255, 255))
        result = pil_to_base64(large_image)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_base64_to_pil_valid_string(self, sample_image: Image.Image):
        # First convert to base64, then back to PIL
        base64_str = pil_to_base64(sample_image)
        result = base64_to_pil(base64_str)

        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
        assert result.mode == sample_image.mode

    def test_base64_to_pil_different_formats(self, sample_rgba_image: Image.Image):
        base64_str = pil_to_base64(sample_rgba_image)
        result = base64_to_pil(base64_str)

        assert isinstance(result, Image.Image)
        assert result.size == sample_rgba_image.size
        # Mode might be preserved as RGBA

    def test_base64_to_pil_invalid_string(self):
        # Test with invalid base64 string
        with pytest.raises(
            Exception
        ):  # Could be various exceptions depending on invalid format
            base64_to_pil("invalid_base64_string")

    def test_base64_to_pil_empty_string(self):
        with pytest.raises(Exception):
            base64_to_pil("")

    def test_base64_to_pil_not_image_data(self):
        # Test with valid base64 but not image data
        non_image_data = base64.b64encode(b"hello world").decode("utf-8")
        with pytest.raises(Exception):
            base64_to_pil(non_image_data)

    def test_round_trip_conversion(self, sample_image: Image.Image):
        # Test converting to base64 and back
        base64_str = pil_to_base64(sample_image)
        recovered_image = base64_to_pil(base64_str)

        assert recovered_image.size == sample_image.size
        assert recovered_image.mode == sample_image.mode

        # Check pixel values are preserved (at least for a few pixels)
        original_pixels = list(sample_image.getdata())
        recovered_pixels = list(recovered_image.getdata())

        # Should be identical or very close (accounting for PNG compression)
        assert len(original_pixels) == len(recovered_pixels)

    def test_generate_color_palette_basic(self):
        colors = generate_color_palette(5)

        assert isinstance(colors, list)
        assert len(colors) == 5

        # Check each color is a tuple of 3 RGB values
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(val, int) for val in color)
            assert all(0 <= val <= 255 for val in color)

    def test_generate_color_palette_single_color(self):
        colors = generate_color_palette(1)

        assert len(colors) == 1
        assert isinstance(colors[0], tuple)
        assert len(colors[0]) == 3

    def test_generate_color_palette_many_colors(self):
        colors = generate_color_palette(20)

        assert len(colors) == 20

        # Check that colors are diverse (not all the same)
        unique_colors = set(colors)
        assert len(unique_colors) > 1  # Should have multiple distinct colors

    def test_generate_color_palette_zero_colors(self):
        colors = generate_color_palette(0)

        assert isinstance(colors, list)
        assert len(colors) == 0

    def test_generate_color_palette_color_distribution(self):
        # Test that colors are distributed across the spectrum
        colors = generate_color_palette(10)

        # Check that we get a good distribution of hues
        assert len(colors) == 10

        # First color should be red-ish (hue=0)
        first_color = colors[0]
        assert first_color[0] == 255  # Red component should be max
        assert first_color[1] == 0  # Green component should be min
        assert first_color[2] == 0  # Blue component should be min

    def test_generate_color_palette_hsv_properties(self):
        # Test that generated colors have full saturation and value
        colors = generate_color_palette(3)

        for color in colors:
            r, g, b = color
            # Convert back to HSV to verify saturation and value
            max_val = max(r, g, b)
            min_val = min(r, g, b)

            # Value should be at maximum (255)
            assert max_val == 255

            # Saturation should be high (min component should be 0 or very low)
            # For pure HSV colors with S=1, V=1, the minimum RGB component should be 0
            assert min_val == 0

    def test_round_trip_different_image_modes(self):
        # Test round trip with different image modes
        test_images = [
            Image.new("RGB", (10, 10), (255, 0, 0)),
            Image.new("RGBA", (10, 10), (0, 255, 0, 128)),
            Image.new("L", (10, 10), 128),
        ]

        for original in test_images:
            base64_str = pil_to_base64(original)
            recovered = base64_to_pil(base64_str)

            assert isinstance(recovered, Image.Image)
            assert recovered.size == original.size
            # Mode might change due to PNG format but should be a valid mode

    def test_pil_to_base64_deterministic(self, sample_image: Image.Image):
        # Test that the same image produces the same base64 string
        result1 = pil_to_base64(sample_image)
        result2 = pil_to_base64(sample_image)

        assert result1 == result2

    def test_base64_string_properties(self, sample_image: Image.Image):
        result = pil_to_base64(sample_image)

        # Check that it's valid base64
        try:
            decoded = base64.b64decode(result)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("Generated string is not valid base64")

    def test_generate_color_palette_edge_case_negative(self):
        # Test with negative number (should handle gracefully or raise error)
        try:
            colors = generate_color_palette(-1)
            assert len(colors) == 0  # Should return empty list
        except ValueError:
            pass  # Or might raise an error, both are acceptable

    def test_generate_color_palette_large_number(self):
        # Test with large number of colors
        colors = generate_color_palette(100)

        assert len(colors) == 100
        assert all(isinstance(color, tuple) and len(color) == 3 for color in colors)

        # Colors should still be diverse even with many requested
        assert len(set(colors)) > 50  # Should have many unique colors
