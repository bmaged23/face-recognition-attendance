"""Decode uploaded photos consistently for enrollment and image recognition."""
import io


def decode_photo(data: bytes, max_bytes: int):
    if not data or len(data) > max_bytes:
        raise ValueError(f"photo must contain between 1 and {max_bytes} bytes")
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError("photo must not exceed 25 million pixels")
            return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))[:, :, ::-1].copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("photo must be a valid image") from exc
