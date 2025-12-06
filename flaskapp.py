import io
import logging
import traceback
import json

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
from baselines.ViT.ViT_explanation_generator import LRP
from baselines.ViT.ViT_new import vit_base_patch16_224 as vit_orig


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

# Load class index → name mapping
with open("cls2idx.json", "r") as f:
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
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    logger.info("Loading ViT_LRP model...")
    model = vit_LRP(pretrained=True).to(device)
    model.eval()
    inference_model = vit_orig(pretrained=True).to(device)
    inference_model.eval()
    logger.info("ViT_LRP models loaded successfully.")
    attribution_generator = LRP(model)
    logger.info("Model and LRP initialized successfully.")
except Exception as e:
    logger.error("Failed to initialize model / LRP:")
    logger.error(e)
    traceback.print_exc()
    raise e  # fail fast so you notice


def compute_attribution_map(original_image, class_index=None):
    """
    original_image: tensor [3, 224, 224] (normalized)
    returns: torch.Tensor [224, 224] on CPU, normalized to [0,1]
    """
    # [1, 3, 224, 224] on device, with gradients for LRP
    input_tensor = original_image.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Run LRP
    transformer_attribution = attribution_generator.generate_LRP(
        input_tensor,
        method="transformer_attribution",
        index=class_index
    )  # tensor on device

    # Reshape and upscale to 224x224
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = F.interpolate(
        transformer_attribution,
        scale_factor=16,
        mode="bilinear",
        align_corners=False,
    )
    transformer_attribution = transformer_attribution.reshape(224, 224)

    # Normalize to [0,1]
    min_val = transformer_attribution.min()
    max_val = transformer_attribution.max()
    attr_norm = (transformer_attribution - min_val) / (max_val - min_val + 1e-8)

    # Return on CPU
    return attr_norm.detach().cpu()
# --------------------------
# Visualization function
# --------------------------
def generate_visualization(original_image, class_index=None):
    """
    original_image: tensor [3, 224, 224] (normalized)
    returns: np.array HxWx3 (BGR, uint8)
    """
    # 1) get normalized attribution [224,224] on CPU
    attr = compute_attribution_map(original_image, class_index=class_index)  # torch
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
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)  # to BGR for OpenCV
    return vis

def run_perturbation(img_raw, img_norm, target_index=None,perturbation_type="positive"):
    """
    img_raw: [3,224,224] tensor in [0,1]
    img_norm: [3,224,224] normalized tensor
    target_index: int or None

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
    base_size = 224 * 224
    perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
    attr = compute_attribution_map(img_norm, class_index=target_index_int)  # [224,224]
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
        logger.debug("Generating visualization...")
        vis_bgr = generate_visualization(img_tensor, class_index=class_index)

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
        )

        return jsonify({
            "perturbation_type": perturbation_type,
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

if __name__ == "__main__":
    # Run on localhost:5000
    logger.info("Running Flask app on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
