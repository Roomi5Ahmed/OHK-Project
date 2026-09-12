"""CNN-based water detection using trained U-Net on Sen1Floods11."""

import numpy as np
import torch
import rasterio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.aoi_config import DATA_DIR

_MODEL = None
_MODEL_PATH = Path("C:/Git/OHK Project/models/flood_unet_final.pt")

# Sen1Floods11 normalization stats
S1_MEAN = np.array([0.6851, 0.5235]).reshape(2, 1, 1)
S1_STD = np.array([0.0820, 0.1102]).reshape(2, 1, 1)


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not _MODEL_PATH.exists():
        return None

    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=2, classes=2)
    state = torch.load(_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _MODEL = model
    return model


def _normalize_s1(s1_db):
    """Normalize S1 dB data to match Sen1Floods11 training distribution."""
    s1 = np.nan_to_num(s1_db, nan=-25.0)
    s1 = np.clip(s1, -50, 1)
    s1 = (s1 + 50) / 51.0
    s1 = (s1 - S1_MEAN) / S1_STD
    return s1.astype(np.float32)


def predict_water_mask(s1_vh_db, s1_vv_db, tile_size=256, overlap=64):
    """
    Run U-Net inference on S1 VH+VV dB data.

    Args:
        s1_vh_db: VH band in dB, shape (H, W)
        s1_vv_db: VV band in dB, shape (H, W)
        tile_size: inference tile size (must match training)
        overlap: overlap between tiles to avoid border artifacts

    Returns:
        water_prob: float32 array (H, W), probability of water [0,1]
    """
    model = _load_model()
    if model is None:
        return None

    H, W = s1_vh_db.shape
    s1 = np.stack([s1_vh_db, s1_vv_db], axis=0)  # (2, H, W)
    s1 = _normalize_s1(s1)

    # Pad to multiple of tile_size
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0:
        s1_padded = np.pad(s1, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    else:
        s1_padded = s1

    _, Ph, Pw = s1_padded.shape
    step = tile_size - overlap
    water_prob = np.zeros((Ph, Pw), dtype=np.float32)
    count = np.zeros((Ph, Pw), dtype=np.float32)

    # Sliding window inference
    with torch.no_grad():
        for y in range(0, Ph - tile_size + 1, step):
            for x in range(0, Pw - tile_size + 1, step):
                tile = s1_padded[:, y:y+tile_size, x:x+tile_size]
                tile_tensor = torch.tensor(tile, dtype=torch.float32).unsqueeze(0)
                prob = torch.softmax(model(tile_tensor), dim=1)[0, 1].numpy()
                water_prob[y:y+tile_size, x:x+tile_size] += prob
                count[y:y+tile_size, x:x+tile_size] += 1.0

    count = np.maximum(count, 1.0)
    water_prob /= count

    # Remove padding
    return water_prob[:H, :W].astype(np.float32)


def cnn_water_index(s1_vh_db, s1_vv_db, jrc_water=None):
    """
    CNN-based flood water detection with permanent water subtraction.

    Args:
        s1_vh_db: VH band in dB
        s1_vv_db: VV band in dB
        jrc_water: optional pre-loaded JRC permanent water mask (float32, 0-1)

    Returns:
        flood_water: float32 (H, W), flood-only water probability
    """
    water_prob = predict_water_mask(s1_vh_db, s1_vv_db)
    if water_prob is None:
        return None

    # Subtract permanent water
    if jrc_water is not None:
        flood_water = water_prob * (1 - jrc_water)
    else:
        flood_water = water_prob

    return flood_water.astype(np.float32)
