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
from baselines.ViT.ViT_LRP import vit_base_patch16_224
from baselines.ViT.ViT_explanation_generator import Baselines,LRP
from baselines.ViT.ViT_new import vit_base_patch16_224 as vit_orig

from baselines.ViT.ViT_LRP import vit_base_patch14_reg4_dinov2


# --------------------------
# Config
# --------------------------
use_thresholding = False

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

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
def show_cam_on_image(img, mask):
    """
    img: H x W x 3, float in [0,1]
    mask: H x W, float in [0,1]
    """
    logger.debug(f"show_cam_on_image: img.shape={img.shape}, mask.shape={mask.shape}")
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# --------------------------
# Load model once at startup
# --------------------------
# Custom checkpoint directory
checkpoint_dir = os.path.join(_script_dir, "model_checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
logger.info(f"Using checkpoint directory: {checkpoint_dir}")

import json 

all_model_details_json_list= list(json.load(open("models_details.json")))
all_model_details_dict = dict()
for i, item in enumerate(all_model_details_json_list):
    all_model_details_dict[item["model_id"]] = item


# Track current device for models
_current_device = None
SELECTED_MODEL_ID = None

def get_device(requested_device=None):
    """
    Get and validate device. Move models if needed.
    requested_device: str or None - 'cpu', 'cuda', or 'mps'
    """
    global model, inference_model, baselines, attribution_generator, _current_device
    
    if requested_device is None:
        # Default: use CUDA if available, else CPU
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validate device
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
    
    # Move models to requested device if different
    if _current_device != requested_device:
        logger.info(f"Moving models to device: {requested_device}")
        model = model.to(requested_device)
        inference_model = inference_model.to(requested_device)
        baselines = Baselines(inference_model)
        attribution_generator = LRP(model)
        _current_device = requested_device
    
    return requested_device

# Load models on CPU initially (will be moved to requested device on first use)
try:

    if SELECTED_MODEL_ID == "vit_base_patch16_224.augreg2_in21k_ft_in1k":
        logger.info(f"Loading {SELECTED_MODEL_ID} ViT LRP model...")
        model = vit_base_patch16_224(pretrained=True, checkpoint_dir=checkpoint_dir).to("cpu")
        model.eval()
        inference_model = vit_orig(pretrained=True, checkpoint_dir=checkpoint_dir).to("cpu")
        inference_model.eval()
        baselines = Baselines(inference_model)
        logger.info("ViT_LRP models loaded successfully.")
        attribution_generator = LRP(model)
        _current_device = "cpu"
        logger.info("Model and LRP initialized successfully (on CPU, will move to requested device on first use).")
    
    elif SELECTED_MODEL_ID == "vit_base_patch14_reg4_dinov2":
        logger.info(f"Loading {SELECTED_MODEL_ID} ViT LRP model...")

        model_details_dict = all_model_details_dict[SELECTED_MODEL_ID]
        model = vit_base_patch14_reg4_dinov2(
            pretrained = False, 
            checkpoint_dir = model_details_dict["finetuned_head_path"]
        ).to("cpu")
        model.eval()
        inference_model = vit_orig(pretrained=True, checkpoint_dir=checkpoint_dir).to("cpu")
        inference_model.eval()
        baselines = Baselines(inference_model)
        logger.info("ViT_LRP models loaded successfully.")
        attribution_generator = LRP(model)
        _current_device = "cpu"
        logger.info("Model and LRP initialized successfully (on CPU, will move to requested device on first use).")
        
        
        except Exception as e:
    logger.error("Failed to initialize model / LRP:")
    logger.error(e)
    traceback.print_exc()
    raise e  # fail fast so you notice



def compute_attribution_map(original_image, class_index=None, method="transformer_attribution", device="cpu"):
    """
    original_image: tensor [3, 224, 224] (normalized)
    class_index: int or None
    method: one of VALID_ATTR_METHODS
    device: str - device to use ('cpu', 'cuda', 'mps')
    returns: torch.Tensor [224, 224] on CPU, in [0,1]
    """
    # Get and set device
    device = get_device(device)

    # [1, 3, 224, 224] on device, with gradients for LRP
    input_tensor = original_image.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Run LRP
    try:
        if SELECTED_MODEL_ID == "vit_base_patch16_224.augreg2_in21k_ft_in1k":
            from pprint import pprint
            pprint(all_model_details_dict[SELECTED_MODEL_ID])
            
            if method == "attn_gradcam":
                transformer_attribution = baselines.generate_cam_attn(input_tensor, index=class_index)
            else:
                transformer_attribution = attribution_generator.generate_LRP(
                    input_tensor,
                    method=method,
                    index=class_index
                )  # tensor on device
        
        elif SELECTED_MODEL_ID == "vit_base_patch14_reg4_dinov2":

            model_lrp = vit_base_patch14_reg4_dinov2(pretrained=False, img_size=518).to(device)

            model_details = all_model_details_dict[SELECTED_MODEL_ID]
            
            if method == "attn_gradcam":
                transformer_attribution = baselines.generate_cam_attn(input_tensor, index=class_index)
            else:
                transformer_attribution = attribution_generator.generate_LRP(
                    input_tensor,
                    method=method,
                    index=class_index
                )  # tensor on device


        # TODO: Add all other models LRP here
        else:
            raise ValueError(f"Invalid model ID: {SELECTED_MODEL_ID}")
        logger.debug(f"transformer_attribution shape after generate_LRP: {transformer_attribution.shape}, dims: {transformer_attribution.dim()}")
        
        # Handle different return shapes from generate_LRP
        # Ensure we have a 4D tensor [B, C, H, W] for interpolation
        if transformer_attribution.dim() == 0:
            raise ValueError(f"Received scalar tensor from generate_LRP (method={method})")
        elif transformer_attribution.dim() == 1:
            # 1D tensor - try to reshape to 2D square
            numel = transformer_attribution.numel()
            size = int(np.sqrt(numel))
            if size * size == numel:
                transformer_attribution = transformer_attribution.reshape(size, size)
            else:
                raise ValueError(f"Cannot reshape 1D tensor of size {numel} to square")
        
        # Now ensure 2D or higher
        if transformer_attribution.dim() == 2:
            # Add batch and channel dims: [H, W] -> [1, 1, H, W]
            transformer_attribution = transformer_attribution.unsqueeze(0).unsqueeze(0)
        elif transformer_attribution.dim() == 3:
            # [C, H, W] or [B, H, W] -> [1, 1, H, W]
            if transformer_attribution.shape[0] == 1:
                transformer_attribution = transformer_attribution.unsqueeze(0)
            else:
                # Take first channel or average
                transformer_attribution = transformer_attribution.mean(dim=0, keepdim=True).unsqueeze(0)
        elif transformer_attribution.dim() == 4:
            # Already 4D, but ensure it's [1, 1, H, W]
            if transformer_attribution.shape[0] != 1:
                transformer_attribution = transformer_attribution[0:1]
            if transformer_attribution.shape[1] != 1:
                transformer_attribution = transformer_attribution.mean(dim=1, keepdim=True)
        
        logger.debug(f"transformer_attribution shape after normalization: {transformer_attribution.shape}")
        
        # Reshape and upscale based on method
        if method == "full":
            # Full LRP should already be 224x224 or close
            if transformer_attribution.shape[-2:] != (224, 224):
                transformer_attribution = F.interpolate(
                    transformer_attribution,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            # Other methods: first ensure 14x14, then upscale to 224x224
            current_h, current_w = transformer_attribution.shape[-2:]
            if (current_h, current_w) != (14, 14):
                # Try to reshape if total elements match
                if current_h * current_w == 14 * 14:
                    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
                else:
                    # Interpolate to 14x14
                    transformer_attribution = F.interpolate(
                        transformer_attribution,
                        size=(14, 14),
                        mode="bilinear",
                        align_corners=False,
                    )
            
            # Upscale from 14x14 to 224x224
            transformer_attribution = F.interpolate(
                transformer_attribution,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
        
        # Final reshape to [224, 224]
        transformer_attribution = transformer_attribution.squeeze()
        if transformer_attribution.dim() != 2:
            transformer_attribution = transformer_attribution.reshape(224, 224)
            
    except Exception as e:
        logger.error(f"Error processing attribution map: {e}")
        logger.error(f"Method: {method}, tensor shape: {transformer_attribution.shape if 'transformer_attribution' in locals() else 'N/A'}")
        raise

    # Normalize to [0,1]
    min_val = transformer_attribution.min()
    max_val = transformer_attribution.max()
    attr_norm = (transformer_attribution - min_val) / (max_val - min_val + 1e-8)

    # Return on CPU
    return attr_norm.detach().cpu()
# --------------------------
# Visualization function
# --------------------------
def generate_visualization(original_image, class_index=None, method="transformer_attribution", device="cpu"):
    """
    original_image: tensor [3, 224, 224] (normalized)
    device: str - device to use ('cpu', 'cuda', 'mps')
    returns: np.array HxWx3 (BGR, uint8)
    """
    # 1) get normalized attribution [224,224] on CPU
    attr = compute_attribution_map(original_image, class_index=class_index, method=method, device=device)  # torch
    transformer_attribution = attr.numpy()  # [224,224], float32 in [0,1]

    # 2) prepare original image in [0,1] for overlay
    image_transformer_attribution = original_image.permute(1, 2, 0).cpu().numpy()
    img_min = image_transformer_attribution.min()
    img_max = image_transformer_attribution.max()
    image_transformer_attribution = (
        image_transformer_attribution - img_min
    ) / (img_max - img_min + 1e-8)

    # 3) overlay heatmap
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)  # [0,255]
    # vis = cv2.cvtColor(np.array(vis))  # to BGR for OpenCV
    return vis

def run_perturbation(img_raw, img_norm, target_index=None,perturbation_type="positive", method="transformer_attribution", device="cpu"):
    """
    img_raw: [3,224,224] tensor in [0,1]
    img_norm: [3,224,224] normalized tensor
    target_index: int or None
    perturbation_type: str - 'positive' or 'negative'
    method: str - attribution method
    device: str - device to use ('cpu', 'cuda', 'mps')

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
    # Get and set device
    device = get_device(device)
    
    base_size = 224 * 224
    perturbation_steps = [0.01, 0.05, 0.08, 0.1, 0.15, 0.3, 0.35, 0.4, 0.45]

    # --- Original prediction ---
    with torch.no_grad():
        logits = inference_model(img_norm.unsqueeze(0).to(device))  # [1,1000]
        probs = torch.softmax(logits, dim=1)              # [1,1000]
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
    attr = compute_attribution_map(img_norm, class_index=target_index_int, method=method, device=device)  # [224,224]
    vis = attr.view(-1)  # [224*224]
    
    # Positive: perturb highest-attribution pixels
    # Negative: perturb lowest-attribution pixels => use -vis
    if perturbation_type == "negative":
        vis = -vis

    # --- Prepare data ---
    data = img_raw.unsqueeze(0)  # [1,3,224,224]
    org_shape = data.shape       # [B,C,H,W]

    perturbation_results = []

    for frac in perturbation_steps:
        k = int(base_size * frac)

        _data = data.clone()  # [1,3,224,224]

        # top-k indices in attribution
        _, idx = torch.topk(vis, k, dim=-1)  # [k]

        idx_expanded = idx.unsqueeze(0).unsqueeze(0).repeat(
            org_shape[0], org_shape[1], 1
        )  # [1,3,k]

        _data_flat = _data.view(org_shape[0], org_shape[1], -1)  # [1,3,224*224]
        _data_flat.scatter_(-1, idx_expanded, 0.0)

        _data_perturbed = _data_flat.view(org_shape)  # [1,3,224,224]

        # re-normalize before model
        pert_single = _data_perturbed.squeeze(0)      # [3,224,224] in [0,1]
        pert_norm = normalize(pert_single).unsqueeze(0).to(device)  # [1,3,224,224]

        with torch.no_grad():
            logits_p = inference_model(pert_norm)
            probs_p = torch.softmax(logits_p, dim=1)
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
    
    # device selection
    device = request.form.get("device", "cpu")
    if device not in ["cpu", "cuda", "mps"]:
        device = "cpu"
        logger.warning(f"Invalid device requested, using CPU")

    try:
        # Read and preprocess image
        logger.debug("Opening image with PIL...")
        img = Image.open(file.stream).convert("RGB")
        logger.debug(f"Image opened: size={img.size}, mode={img.mode}")

        logger.debug("Applying torchvision transforms...")
        img_tensor = transform(img)
        logger.debug(
            f"Image transformed: shape={tuple(img_tensor.shape)}, "
            f"dtype={img_tensor.dtype}"
        )

        # Optionally, you could pick a specific class_index
        logger.debug(f"Generating visualization on device: {device}...")
        vis_bgr = generate_visualization(img_tensor, class_index=class_index, method=method, device=device)

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
    
    # device selection
    device = request.form.get("device", "cpu")
    if device not in ["cpu", "cuda", "mps"]:
        device = "cpu"
        logger.warning(f"Invalid device requested, using CPU")

    try:
        img = Image.open(file.stream).convert("RGB")
        img_norm = transform(img)            # normalized [3,224,224]
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
    
    # Get and set device (move models if needed)
    device = get_device(device)

    try:
        img = Image.open(file.stream).convert("RGB")
        img_norm = transform(img)  # [3,224,224]

        with torch.no_grad():
            logits = inference_model(img_norm.unsqueeze(0).to(device))  # [1,num_classes]
            probs = torch.softmax(logits, dim=1)

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

@app.route("/api/selected_model", methods=["POST"])
def set_selected_model():
    """
    Set the selected model for the session.
    Expects JSON with {"model_id": "model_id_here"}.
    """
    global SELECTED_MODEL_ID
    logger.info("➡️ /api/selected_model called")
    
    try:
        data = request.get_json()
        if not data or "model_id" not in data:
            return jsonify({"error": "Missing 'model_id' in request body"}), 400
        
        model_id = data["model_id"]
        
        

        SELECTED_MODEL_ID = model_id
        logger.debug(f"Setting global selected model: {SELECTED_MODEL_ID}")
        
        return jsonify({"selected_model": SELECTED_MODEL_ID}), 200
    
    except Exception as e:
        logger.error(f"Error setting selected model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def get_models():
    """
    Returns a list of available models from models_details.json
    """
    logger.info("➡️ /api/models called")
    try:
        details_path = os.path.join(_script_dir, "models_details.json")
        with open(details_path, "r") as f:
            all_models = json.load(f)
        
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
