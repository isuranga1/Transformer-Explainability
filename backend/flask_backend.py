import io
import logging
import traceback
import json
import sys
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

from PIL import Image
import torchvision.transforms as transforms

# --------------------------
# Setup logging (DEBUG mode)
# --------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("Starting Flask + ViT-LRP server...")

# --------------------------
# Your ViT + LRP imports
# --------------------------
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_LRP import vit_base_patch14_reg4_dinov2 as vit_LRP_reg4
from baselines.ViT.ViT_explanation_generator import Baselines,LRP
from baselines.ViT.ViT_new import vit_base_patch16_224 as vit_orig
from baselines.ViT.ViT_new import vit_base_patch14_reg4_dinov2 as vit_orig_reg4


# --------------------------
# Config
# --------------------------
use_thresholding = False

# Transformations (default, will be updated dynamically)
transform = None
current_img_size = 224

def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# Populate initial transform
transform = get_transform(224)

VALID_ATTR_METHODS = [
    "rollout",
    "transformer_attribution",
    "full",
    "last_layer",
    "last_layer_attn",
    "attn_gradcam",
]

_script_dir = os.path.dirname(os.path.abspath(__file__))
# Load class index → name mapping
with open(os.path.join(_script_dir, "cls2idx.json"), "r") as f:
    CLS2IDX = json.load(f)

def get_class_name(idx: int) -> str:
    return CLS2IDX.get(str(idx), f"class_{idx}")
# --------------------------
# Heatmap helper
# --------------------------
def show_cam_on_image(img, mask, use_rgb=False):
    """
    img: H x W x 3, float in [0,1]
    mask: H x W, float in [0,1]
    """
    logger.debug(f"show_cam_on_image: img.shape={img.shape}, mask.shape={mask.shape}")
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Convert heatmap to RGB if img is RGB
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = cam / np.max(cam)
    return cam

def get_models_details_json():
    """Helper to load models_details.json"""
    try:
        details_path = os.path.join(_script_dir, "models_details.json")
        with open(details_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"models_details.json not found at {details_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding models_details.json at {details_path}")
        return []

def get_model_info(model_id):
    models = get_models_details_json()
    for m in models:
        if m["model_id"] == model_id:
            return m
    return None

# --------------------------
# Load model logic
# --------------------------
# Custom checkpoint directory
checkpoint_dir = os.path.join(_script_dir, "model_checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
logger.info(f"Using checkpoint directory: {checkpoint_dir}")

# Global model state
_current_model_id = None
_current_device = None

model = None
inference_model = None
baselines = None
attribution_generator = None

# Initial default model (can be changed via request)
DEFAULT_MODEL_ID = "vit_base_patch16_224.augreg2_in21k_ft_in1k"

def load_models(model_id, device):
    global model, inference_model, baselines, attribution_generator, _current_model_id, _current_device, transform, current_img_size
    
    # Check if we need to reload
    if model_id == _current_model_id and device == _current_device:
        return

    logger.info(f"Loading model '{model_id}' on device '{device}'...")

    # Update transforms based on model details
    model_info = get_model_info(model_id)
    if model_info:
        img_size = model_info.get("input_image_resolution", 224)
        logger.info(f"Setting image resolution to {img_size} for model {model_id}")
        current_img_size = img_size
        transform = get_transform(img_size)
    else:
        logger.warning(f"No metadata found for {model_id}, defaulting to 224")
        current_img_size = 224
        transform = get_transform(224)

    try:
        # TODO: Add specific factory calls for other models (deit3, etc.)
        # For now, we attempt to use the same class if possible, or fallback
        # if model_id != DEFAULT_MODEL_ID and "vit_base_patch16_224" == model_id:
        if model_id != DEFAULT_MODEL_ID and "vit_base_patch14_reg4_dinov2" not in model_id and "vit_base_patch16" not in model_id and "deit3_base_patch16" not in model_id:
             logger.warning(f"Model '{model_id}' is not fully supported yet. Falling back to default architecture structure.")

        if "vit_base_patch16" in model_id or "deit3_base_patch16" in model_id:
             m_lrp = vit_LRP(pretrained=True, checkpoint_dir=checkpoint_dir)
             m_orig = vit_orig(pretrained=True, checkpoint_dir=checkpoint_dir)
        elif "vit_base_patch14_reg4_dinov2" in model_id:
             # This model likely needs 518 resolution, need to check if we handle transforms for it too?
             # For now, just load the model.
             # Use pretrained=False and load weights manually as per user pattern
             m_lrp = vit_LRP_reg4(pretrained=False, checkpoint_dir=checkpoint_dir, img_size=current_img_size)
             m_orig = vit_orig_reg4(pretrained=False, checkpoint_dir=checkpoint_dir, img_size=current_img_size)
             
             finetuned_path = os.path.join(checkpoint_dir, f"{model_id}_finetuned_best.pth")
             # Try custom name from models_details.json if available
             if model_info and model_info.get("finetuned_head_path") and model_info.get("finetuned_head_path") != "-":
                 finetuned_path = os.path.join(checkpoint_dir, model_info.get("finetuned_head_path"))

             if os.path.exists(finetuned_path):
                 logger.info(f"Loading fine-tuned weights from {finetuned_path}")
                 state_dict = torch.load(finetuned_path, map_location=device)
                 m_lrp.load_state_dict(state_dict, strict=False)
                 m_orig.load_state_dict(state_dict, strict=False)
             else:
                 logger.warning(f"Fine-tuned weights not found at {finetuned_path}. Using random/timm init.")

        else:
             # Fallback
             m_lrp = vit_LRP(pretrained=True, checkpoint_dir=checkpoint_dir)
             m_orig = vit_orig(pretrained=True, checkpoint_dir=checkpoint_dir)

        m_lrp = m_lrp.to(device)
        m_lrp.eval()
        
        m_orig = m_orig.to(device)
        m_orig.eval()
        
        model = m_lrp
        inference_model = m_orig
        baselines = Baselines(inference_model)
        attribution_generator = LRP(model)
        
        _current_model_id = model_id
        _current_device = device
        
        logger.info(f"Model '{model_id}' loaded successfully on '{device}'.")
        
    except Exception as e:
        logger.error(f"Failed to load model '{model_id}': {e}")
        traceback.print_exc()
        raise e

def get_device(requested_device=None):
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        requested_device = "cpu"
    elif requested_device == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            requested_device = "mps"
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            requested_device = "cpu"
    elif requested_device not in ["cpu", "cuda", "mps"]:
        logger.warning(f"Invalid device '{requested_device}', falling back to CPU")
        requested_device = "cpu"
    return requested_device

# Initialize default
try:
    load_models(DEFAULT_MODEL_ID, "cpu")
except Exception as e:
    logger.error(f"Startup model load failed: {e}")


def compute_attribution_map(original_image, class_index=None, method="transformer_attribution", device="cpu", model_id=DEFAULT_MODEL_ID):
    """
    original_image: tensor [3, H, W] (normalized)
    class_index: int or None
    method: one of VALID_ATTR_METHODS
    device: str - device to use ('cpu', 'cuda', 'mps')
    model_id: str - model to use
    returns: torch.Tensor [H, W] on CPU, in [0,1]
    """
    # Get and set device / model
    device = get_device(device)
    load_models(model_id, device)

    # [1, 3, H, W] on device, with gradients for LRP
    input_tensor = original_image.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Run LRP
    try:
        if method == "attn_gradcam":
            transformer_attribution = baselines.generate_cam_attn(input_tensor, index=class_index)
        else:
            # returns (1, 1, 14, 14) or similar grid
            transformer_attribution = attribution_generator.generate_LRP(
                input_tensor, method=method, index=class_index
            ).detach()

        # Check dimension and reshape if necessary (e.g. from [1, 196] to [1, 1, 14, 14])
        logger.debug(f"Shape before reshape check: {transformer_attribution.shape}")
        if transformer_attribution.dim() == 2:
            # Assume [B, num_patches]
            b, n = transformer_attribution.shape
            side = int(n ** 0.5)
            logger.debug(f"Reshaping 2D tensor: b={b}, n={n}, side={side}, side*side={side*side}")
            if side * side == n:
                transformer_attribution = transformer_attribution.reshape(b, 1, side, side)
                logger.debug(f"Reshaped to: {transformer_attribution.shape}")
            else:
                 logger.warning(f"Unable to reshape tensor of shape {transformer_attribution.shape} to square grid")

        elif transformer_attribution.dim() == 3:
             # scale [B, H, W] -> [B, 1, H, W]
             if transformer_attribution.shape[1] != 1:
                 transformer_attribution = transformer_attribution.unsqueeze(1)
        
        logger.debug(f"Shape before interpolate: {transformer_attribution.shape}")
        
        # 3) Interpolate back to original resolution (usually 224 or 518)
        # We want to return 224x224 for the UI standard, or should we match current_img_size?
        # The UI probably expects 224x224 layout or flexible.
        # Let's upscale to the *tensor's* spatial size to match the image being visualized.
        target_h, target_w = original_image.shape[-2:] # H, W
        
        if method == "rollout":
             # Rollout might return [1, 1, grid, grid]
             pass

        elif method == ""
        else:
            # Standard LRP methods return [1, 1, grid, grid]
            pass
        
        # Simple interpolation logic
        transformer_attribution = F.interpolate(
            transformer_attribution,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
            
    except Exception as e:
        logger.error(f"Error processing attribution map: {e}")
        logger.error(f"Method: {method}, tensor shape: {transformer_attribution.shape if 'transformer_attribution' in locals() else 'N/A'}")
        raise

    # Normalize to [0,1]
    # transformer_attribution is [1, 1, H, W]
    min_val = transformer_attribution.min()
    max_val = transformer_attribution.max()
    attr_norm = (transformer_attribution - min_val) / (max_val - min_val + 1e-8)

    # Return on CPU, squeezed to [H, W]
    return attr_norm[0, 0].detach().cpu()
# --------------------------
# Visualization function
# --------------------------
def generate_visualization(original_image, class_index=None, method="transformer_attribution", device="cpu", model_id=DEFAULT_MODEL_ID):
    """
    original_image: tensor [3, H, W] (normalized)
    device: str - device to use ('cpu', 'cuda', 'mps')
    model_id: str - model to use
    returns: np.array HxWx3 (BGR, uint8)
    """
    # 1) get normalized attribution [H, W] on CPU
    attr = compute_attribution_map(original_image, class_index=class_index, method=method, device=device, model_id=model_id)  # torch
    transformer_attribution = attr.numpy()  # [H, W], float32 in [0,1]

    # 2) prepare original image in [0,1] for overlay
    # This expects original_image to be normalized. We need to denorm it or just expect the UI to handle it.
    # Actually `vis.show_cam_on_image` expects `img` to be [H, W, 3] in [0,1].
    
    # We need to un-normalize original_image to display it properly
    # Assuming (0.5, 0.5, 0.5) stats for now
    img_disp = original_image.permute(1, 2, 0).cpu().numpy()
    img_disp = (img_disp * 0.5) + 0.5
    img_disp = np.clip(img_disp, 0, 1)

    # 3) blend
    # Note: `transformer_attribution` might be 518x518, `img_disp` 518x518.
    # Vis library usually handles matching sizes, but we interpolated attr to match img inside compute_attribution_map.
    vis = show_cam_on_image(img_disp, transformer_attribution, use_rgb=True)
    
    # 4) Convert RGB -> BGR for consistency if OpenCV-like behavior is expected downstream or by frontend
    # But wait, Flask sends JPEG/PNG. `show_cam_on_image` returns RGB uint8.
    # If the frontend expects RGB, we are good. 
    # The previous code had `vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)` maybe?
    # Let's check previous implementation pattern.
    vis = np.uint8(255 * vis) # Scale to 0-255
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # standard OpenCV format
    return vis

def run_perturbation(img_raw, img_norm, target_index=None,perturbation_type="positive", method="transformer_attribution", device="cpu", model_id=DEFAULT_MODEL_ID):
    """
    img_raw: [3,H,W] tensor in [0,1]
    img_norm: [3,H,W] normalized tensor
    target_index: int or None
    perturbation_type: str - 'positive' or 'negative'
    method: str - attribution method
    device: str - device to use ('cpu', 'cuda', 'mps')
    model_id: str - model to use

    returns:
        target_class_idx
        target_class_name
        original_prediction: {
           "top1": {...},
           "target": {...}
        }
        perturbation_results: list of {
           "fraction", "top1_class_idx", "top1_class_name",
           "top1_prob", "target_prob"
        }
    """
    # Get and set device / model
    device = get_device(device)
    load_models(model_id, device)
    
    base_size = current_img_size * current_img_size
    perturbation_steps = [0.01, 0.05, 0.08, 0.1, 0.15, 0.3, 0.35, 0.4, 0.45]
    perturbation_results = []
    
    # Initial Prediction
    with torch.no_grad():
        output = inference_model(img_norm.unsqueeze(0).to(device))  # [1,1000]
        probs = torch.softmax(output, dim=1)              # [1,1000]
        top1_prob, top1_idx = torch.max(probs, dim=1)     # [1]

    top1_idx_int = int(top1_idx.item())
    top1_prob_float = float(top1_prob.item())

    # If no target specified, use top-1 as target
    if target_index is None:
        target_index = top1_idx_int

    target_index_int = int(target_index)
    target_prob_float = float(probs[0, target_index_int].item())

    original_prediction = {
        "top1": {
            "class_idx": top1_idx_int,
            "class_name": get_class_name(top1_idx_int),
            "prob": top1_prob_float,
        },
        "target": {
            "class_idx": target_index_int,
            "class_name": get_class_name(target_index_int),
            "prob": target_prob_float,
        },
    }

    # --- Attribution used for perturbation (for target class) ---
    attr = compute_attribution_map(img_norm, class_index=target_index_int, method=method, device=device, model_id=model_id)  # [H,W]
    vis = attr.view(-1)  # [H*W]
    
    # Positive: perturb highest-attribution pixels
    # Negative: perturb lowest-attribution pixels => use -vis
    if perturbation_type == "negative":
        vis = -vis

    # --- Prepare data ---
    data = img_raw.unsqueeze(0)  # [1,3,H,W]
    org_shape = data.shape       # [B,C,H,W]

    for frac in perturbation_steps:
        k = int(base_size * frac)

        _data = data.clone()  # [1,3,H,W]

        # top-k indices in attribution
        _, idx = torch.topk(vis, k, dim=-1)  # [k]

        idx_expanded = idx.unsqueeze(0).unsqueeze(0).repeat(
            org_shape[0], org_shape[1], 1
        )  # [1,3,k]

        _data_flat = _data.view(org_shape[0], org_shape[1], -1)  # [1,3,H*W]
        _data_flat.scatter_(-1, idx_expanded, 0.0)

        _data_perturbed = _data_flat.view(org_shape)  # [1,3,H,W]

        # re-normalize before model
        pert_single = _data_perturbed.squeeze(0)      # [3,H,W] in [0,1]
        pert_norm = transform(transforms.ToPILImage()(pert_single)).unsqueeze(0).to(device) # Re-apply transform to ensure correct normalization

        with torch.no_grad():
            output_p = inference_model(pert_norm)
            probs_p = torch.softmax(output_p, dim=1)
            top1_prob_p, top1_idx_p = torch.max(probs_p, dim=1)
            top1_idx_p_int = int(top1_idx_p.item())
            top1_prob_p_float = float(top1_prob_p.item())
            target_prob_p = float(probs_p[0, target_index_int].item())

        perturbation_results.append({
            "fraction": float(frac),
            "top1_class_idx": top1_idx_p_int,
            "top1_class_name": get_class_name(top1_idx_p_int),
            "top1_prob": top1_prob_p_float,
            "target_prob": target_prob_p,
        })

    return (
        target_index_int,
        get_class_name(target_index_int),
        original_prediction,
        perturbation_results,
    )


# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)
# Allow requests from Next.js dev server (localhost:3000)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/heatmap", methods=["POST"])
def heatmap():
    logger.info("➡️ /api/heatmap called")

    logger.debug(f"request.files keys: {list(request.files.keys())}")
    logger.debug(f"request.form keys: {list(request.form.keys())}")

    if "image" not in request.files:
        logger.warning("No 'image' field in request.files")
        return jsonify({"error": "No image field in form-data (expected 'image')"}), 400

    file = request.files["image"]
    logger.info(
        f"Received file: filename='{file.filename}', content_type='{file.content_type}'"
    )

    if file.filename == "":
        logger.warning("Empty filename in uploaded file.")
        return jsonify({"error": "Empty filename"}), 400
    
    # optional target index
    target_index = request.form.get("target_index", None)
    class_index = None
    if target_index not in (None, ""):
        try:
            class_index = int(target_index)
        except ValueError:
            return jsonify({"error": "target_index must be an integer"}), 400
        
    # attribution method
    method = request.form.get("method", "transformer_attribution")
    if method not in VALID_ATTR_METHODS:
        return jsonify({"error": f"Invalid method '{method}'. Must be one of: {', '.join(VALID_ATTR_METHODS)}"}), 400
    
    # If the user uploads an image, we resize it to the CURRENT model's resolution
    # But wait, the previous cells defined `transform` globally but `load_models` updates it.
    # We need to make sure we use the *current* transform for the *current* model.
    # However, model loading happens *after* we might parse the request?
    # No, we should load model first.
    
    # Extract model_id FIRST
    model_id = request.form.get("model_id", DEFAULT_MODEL_ID)
    
    # device selection
    device = request.form.get("device", "cpu")
    if device not in ["cpu", "cuda", "mps"]:
        device = "cpu"
        logger.warning(f"Invalid device requested, using CPU")

    # Ensure model is loaded (and global transform updated)
    load_models(model_id, device)

    try:
        # Read and preprocess image using the UPDATED transform
        logger.debug("Opening image with PIL...")
        img = Image.open(file.stream).convert("RGB")
        logger.debug(f"Image opened: size={img.size}, mode={img.mode}")

        logger.debug(f"Applying torchvision transforms (size={current_img_size})...")
        img_tensor = transform(img)
        logger.debug(
            f"Image transformed: shape={tuple(img_tensor.shape)}, "
            f"dtype={img_tensor.dtype}"
        )

        # Optionally, you could pick a specific class_index
        logger.debug(f"Generating visualization on device: {device}, model: {model_id}...")
        vis_bgr = generate_visualization(img_tensor, class_index=class_index, method=method, device=device, model_id=model_id)

        logger.debug(
            f"Visualization generated: vis_bgr.shape={vis_bgr.shape}, "
            f"dtype={vis_bgr.dtype}"
        )

        # Convert to PNG in memory
        logger.debug("Converting BGR numpy array to RGB PIL Image...")
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        vis_pil = Image.fromarray(vis_rgb)

        buf = io.BytesIO()
        vis_pil.save(buf, format="PNG")
        buf.seek(0)

        logger.info("✅ Successfully generated heatmap, sending PNG response.")
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        logger.error("🔥 Exception in /api/heatmap")
        logger.error(e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/perturbation", methods=["POST"])
def perturbation():
    if "image" not in request.files:
        return jsonify({"error": "No image field in form-data (expected 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # optional target index
    target_index_raw = request.form.get("target_index", None)
    target_index = None
    if target_index_raw not in (None, ""):
        try:
            target_index = int(target_index_raw)
        except ValueError:
            return jsonify({"error": "target_index must be an integer"}), 400

    # perturbation type: "positive" (default) or "negative"
    perturbation_type = request.form.get("perturbation_type", "positive").lower()
    if perturbation_type not in ("positive", "negative"):
        return jsonify({"error": "perturbation_type must be 'positive' or 'negative'"}), 400
    
    # attribution method
    method = request.form.get("method", "transformer_attribution")
    if method not in VALID_ATTR_METHODS:
        return jsonify({"error": f"Invalid method '{method}'. Must be one of: {', '.join(VALID_ATTR_METHODS)}"}), 400
    
    # model selection
    model_id = request.form.get("model_id", DEFAULT_MODEL_ID)
    
    # device selection
    device = request.form.get("device", "cpu")
    if device not in ["cpu", "cuda", "mps"]:
        device = "cpu"
        logger.warning(f"Invalid device requested, using CPU")

    # Load model & update transform
    load_models(model_id, device)

    try:
        img = Image.open(file.stream).convert("RGB")
        img_norm = transform(img)            # normalized [3,H,W]
        img_raw = img_norm * 0.5 + 0.5       # undo norm → [0,1]

        (
            target_idx,
            target_name,
            original_prediction,
            perturbation_results,
        ) = run_perturbation(
            img_raw,
            img_norm,
            target_index=target_index,
            perturbation_type=perturbation_type,
            method=method,
            device=device,
            model_id=model_id,
        )

        return jsonify({
            "perturbation_type": perturbation_type,
            "method": method,
            "target_class_idx": target_idx,
            "target_class_name": target_name,
            "original_prediction": original_prediction,
            "perturbation_results": perturbation_results,
        })

    except Exception as e:
        print("Error in /api/perturbation:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/infer", methods=["POST"])
def infer():
    """
    Run plain classification on a given image (original or perturbed).
    Returns top-5 predictions with class names and probabilities.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in form-data (expected 'image')"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # device selection
    device = request.form.get("device", "cpu")
    if device not in ["cpu", "cuda", "mps"]:
        device = "cpu"
        logger.warning(f"Invalid device requested, using CPU")
    
    model_id = request.form.get("model_id", DEFAULT_MODEL_ID)

    # Get and set device (move models if needed)
    device = get_device(device)
    load_models(model_id, device)

    try:
        img = Image.open(file.stream).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = inference_model(img_t)  # [1,num_classes]
            probs = torch.softmax(output, dim=1)

        # Top-5
        topk = 5
        top_probs, top_idxs = torch.topk(probs, k=topk, dim=1)
        top_probs = top_probs[0].cpu().tolist()
        top_idxs = top_idxs[0].cpu().tolist()

        predictions = []
        for idx, p in zip(top_idxs, top_probs):
            predictions.append({
                "class_idx": int(idx),
                "class_name": get_class_name(int(idx)),
                "prob": float(p),
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        logger.error("Error in /api/infer:", e)
        return jsonify({"error": str(e)}), 500



@app.route("/api/models", methods=["GET"])
def list_models():
    """
    Returns a list of available models from models_details.json
    """
    logger.info("➡️ /api/models called")
    try:
        all_models = get_models_details_json()
        
        # Filter for filtering available models if needed, 
        # or return all and let frontend decide?
        # User request: "model_id should appear in the dropdown ... only if 'available' is true"
        # I can filter here or in frontend. Filtering here reduces payload.
        
        available_models = [m for m in all_models if m.get("available") is True]
        
        return jsonify({"models": available_models})
    except Exception as e:
        logger.error(f"Error reading models_details.json: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Flask backend for ViT explainability")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask server on (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the Flask server to (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", default=True, help="Run Flask in debug mode (default: True)")
    args = parser.parse_args()
    
    logger.info(f"Running Flask app on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
