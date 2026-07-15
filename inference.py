import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
# Import the official Hugging Face class to map your specific ViT checkpoint keys smoothly
from transformers import ViTForImageClassification

# Import our configurations
import config
# Import our blueprints (keeping convnext as is)
from model import build_convnext_custom_head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple helper function in case crop is enabled in config
def crop_text_area_pil(im):
    # If you have a real cropping implementation, put it here. Otherwise:
    return im

# Pipeline transformations configuration
val_transform = transforms.Compose([
    transforms.Lambda(lambda im: crop_text_area_pil(im) if config.USE_CROP else im),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
])

def load_ensemble_models(version: str):
    """
    Downloads model checkpoints from Hugging Face Hub, builds the structural 
    architectures, loads the weights, and prepares them for eval inference.
    """
    vit_filename = config.MODEL_FILES[version][0]
    cnext_filename = config.MODEL_FILES[version][1]

    print(f"📥 Downloading files for {version} from Hugging Face...")
    vit_path = hf_hub_download(repo_id=config.MODEL_REPO, filename=vit_filename)
    cnext_path = hf_hub_download(repo_id=config.MODEL_REPO, filename=cnext_filename)

    # 1. FIX: Swap custom head builder out for official class to eliminate State Dict errors
    print("🧠 Initializing ViT Architecture...")
    vit_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    # Re-apply the custom MLP Sequential classifier head exactly as defined in training
    hidden = vit_model.config.hidden_size
    mid = int(hidden * 0.5)  # Hidden ratio of 0.5 matching model.py
    vit_model.classifier = nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Dropout(0.30),  # Dropout matching model.py
        nn.Linear(hidden, mid),
        nn.GELU(),
        nn.Dropout(0.30),
        nn.Linear(mid, config.NUM_CLASSES),
    )

    # In-memory translation of state dict to handle library-level key changes
    print("🔄 Checking and translating ViT state dict keys...")
    state_dict = torch.load(vit_path, map_location=device)
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        
        # Translate encoder layers structure
        if "vit.encoder.layer" in key:
            # Replace 'encoder.layer.X' with 'layers.X'
            new_key = new_key.replace("vit.encoder.layer", "vit.layers")
            
            # Map legacy attention to modern attention projection names
            new_key = new_key.replace("attention.attention.query", "attention.q_proj")
            new_key = new_key.replace("attention.attention.key", "attention.k_proj")
            new_key = new_key.replace("attention.attention.value", "attention.v_proj")
            new_key = new_key.replace("attention.output.dense", "attention.o_proj")
            
            # Map legacy intermediate/output MLP to modern fc1/fc2 names
            new_key = new_key.replace("intermediate.dense", "mlp.fc1")
            new_key = new_key.replace("output.dense", "mlp.fc2")
            
        new_state_dict[new_key] = value

    vit_model.load_state_dict(new_state_dict, strict=True)
    vit_model.to(device).eval()

    # 2. Build & Load ConvNeXt
    print("🧠 Initializing ConvNeXt Architecture...")
    cnext_model = build_convnext_custom_head(
        num_classes=config.NUM_CLASSES, 
        variant=config.CONVNEXT_VARIANT, 
        dropout=0.0,
        pretrained=False
    )
    cnext_model.load_state_dict(torch.load(cnext_path, map_location=device), strict=True)
    cnext_model.to(device).eval()

    print("✅ Ensemble loaded successfully into system RAM!")
    return vit_model, cnext_model


@torch.no_grad()
def predict_image(image_path, version: str):
    """
    Runs an incoming image (either a string path or a PIL Image object) 
    through both models, blends their probabilities, and outputs the result.
    """
    vit_model, cnext_model = load_ensemble_models(version)
    
    # --- FIX START: Handle both file paths and pre-opened PIL Images ---
    if isinstance(image_path, Image.Image):
        # If it's already a PIL Image, just copy and convert it to RGB
        img = image_path.convert("RGB")
    else:
        # Otherwise, open it from the path/file stream
        img = Image.open(image_path).convert("RGB")
    # --- FIX END ---

    tensor_img = val_transform(img).unsqueeze(0).to(device) # Shape: [1, 3, 224, 224]

    # Collect logits outputs
    vit_out = vit_model(tensor_img)
    vit_logits = vit_out.logits if hasattr(vit_out, "logits") else vit_out
    cnext_logits = cnext_model(tensor_img)

    # Convert logits arrays to clean probability spectrum distributions
    vit_probs = F.softmax(vit_logits, dim=-1)
    cnext_probs = F.softmax(cnext_logits, dim=-1)

    # Apply the soft-voting blend equation formula
    w_vit, w_cnx = config.VOTING_WEIGHTS[0], config.VOTING_WEIGHTS[1]
    final_probs = (w_vit * vit_probs) + (w_cnx * cnext_probs)

    # Compute final metrics results
    predicted_class = torch.argmax(final_probs, dim=-1).item()
    confidence_score = final_probs[0, predicted_class].item()
    
    predicted_class_name = config.CLASS_NAMES[predicted_class]

    return {
        "class_id": predicted_class,
        "class_name": predicted_class_name,
        "confidence": confidence_score
    }
