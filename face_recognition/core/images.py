"""Decode uploaded photos consistently for enrollment and image recognition."""
import io


def decode_photo(data: bytes, max_bytes: int, max_dimension: int = 1600):
    if not data or len(data) > max_bytes:
        raise ValueError(f"photo must contain between 1 and {max_bytes} bytes")
    if max_dimension <= 0:
        raise ValueError('max_dimension must be positive')
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError("photo must not exceed 25 million pixels")
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((max_dimension, max_dimension))
            return np.asarray(image)[:, :, ::-1].copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("photo must be a valid image") from exc
