import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from diffusers import StableDiffusionPipeline
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

try:
    import lpips
except ImportError:
    raise ImportError("Please install the lpips package: pip install lpips")

METADATA_CSV         = "../processing/train_data/metadata.csv"           
IMAGE_ROOT           = "../../data/images/"                         
NUM_ENTRIES          = 50                        
LORA_WEIGHTS_DIR     = "./fantasy_map_lora/checkpoint-1500/"
LORA_WEIGHT_FILENAME = "pytorch_lora_weights.safetensors"
BASE_MODEL           = "runwayml/stable-diffusion-v1-5"
GENERATED_DIR        = "generated"
METRICS_CSV          = "sb_metrics.csv"

os.makedirs(GENERATED_DIR, exist_ok=True)
os.environ['HF_HOME'] = 'F:\\huggingface_cache'

device = "cuda" if torch.cuda.is_available() else "mps"

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to(device)

pipe.load_lora_weights(LORA_WEIGHTS_DIR, weight_name=LORA_WEIGHT_FILENAME)
pipe.fuse_lora()

lpips_model = lpips.LPIPS(net='alex').to(device)
def load_image_as_tensor(path, resolution=(512, 512)):
    """
    1. Loads a PIL image from `path`.
    2. Resizes it to `resolution`.
    3. Converts to a torch‐tensor in [-1,1] (shape [1,3,H,W], float32).
    Returns:
      - PIL.Image (RGB)
      - torch.Tensor
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(resolution, resample=Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor = (img_tensor * 2) - 1
    return img, img_tensor

def is_black_placeholder(pil_img):
    """
    Returns True if `pil_img` is essentially a solid black image
    (i.e. likely the NSFW‐filter fallback). We check if all pixels are zero.
    """
    arr = np.array(pil_img)
    # If R=G=B=0 everywhere → pure black
    return np.all(arr == 0)

def compute_lpips(img_tensor_a, img_tensor_b):
    """
    Compute LPIPS distance between two torch‐tensors in [-1,1].
    If one of them is None, returns np.nan.
    """
    try:
        with torch.no_grad():
            dist = lpips_model(img_tensor_a, img_tensor_b)
        return float(dist.item())
    except Exception:
        return np.nan

def compute_ssim_psnr(real_img, gen_img):
    """
    Computes SSIM and PSNR between two PIL Images (same size).
    - Uses `channel_axis=2` instead of `multichannel=True` for recent skimage.
    - If width/height < 7, or any error occurs, returns (np.nan, np.nan).
    """
    try:
        arr_a = np.array(real_img).astype(np.float32) / 255.0
        arr_b = np.array(gen_img).astype(np.float32) / 255.0

        h, w = arr_a.shape[:2]
        if min(h, w) < 7:
            return np.nan, np.nan

        ps = psnr(arr_a, arr_b, data_range=1.0)
        ss, _ = ssim(arr_a, arr_b, data_range=1.0, channel_axis=2, full=True)
        return float(ss), float(ps)
    except Exception:
        return np.nan, np.nan

def compute_edge_density(img):
    """
    Convert a PIL.Image to grayscale, apply Laplacian filter,
    and return sum of absolute Laplacian responses.
    If `img` is None or any error, returns np.nan.
    """
    try:
        gray = np.array(img.convert("L"))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.sum(np.abs(lap)))
    except Exception:
        return np.nan

df = pd.read_csv(METADATA_CSV)

records = []

for idx, row in df.head(NUM_ENTRIES).iterrows():
    try:
        file_path = os.path.join(IMAGE_ROOT, row["file_name"])
        prompt = row["label"]
        real_img, real_tensor = load_image_as_tensor(file_path)


        with torch.autocast(device):
            gen_result = pipe(prompt, num_inference_steps=20)
        gen_img_pil = gen_result.images[0]

        if is_black_placeholder(gen_img_pil):
            records.append({
                "index":        idx,
                "file_name":    row["file_name"],
                "prompt":       prompt,
                "generated_path": None,
                "lpips":        np.nan,
                "ssim":         np.nan,
                "psnr":         np.nan,
                "edge_density_real": float(compute_edge_density(real_img)),
                "edge_density_gen":  np.nan
            })
            print(f"[{idx+1}/{NUM_ENTRIES}] NSFW black fallback detected for prompt: “{prompt}”. Metrics = NaN.")
            continue

        # 5. Otherwise, save the generated image to disk
        generated_path = os.path.join(GENERATED_DIR, f"gen_{idx+1:03d}.png")
        gen_img_pil.save(generated_path)

        # 6. Resize generated image to match real_image size
        #    (real_img was already 512×512)
        gen_img = gen_img_pil.resize(real_img.size, resample=Image.LANCZOS)
        gen_tensor = (torch.from_numpy(np.array(gen_img).astype(np.float32)/255.0)
                    .permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1)

        # 7. Compute LPIPS (real_tensor vs. gen_tensor)
        lpips_val = compute_lpips(real_tensor, gen_tensor)

        # 8. Compute SSIM & PSNR
        ssim_val, psnr_val = compute_ssim_psnr(real_img, gen_img)

        # 9. Compute Edge Density for both real vs. generated
        real_ed = compute_edge_density(real_img)
        gen_ed  = compute_edge_density(gen_img)

        records.append({
            "index":             idx,
            "file_name":         row["file_name"],
            "prompt":            prompt,
            "generated_path":    generated_path,
            "lpips":             lpips_val,
            "ssim":              ssim_val,
            "psnr":              psnr_val,
            "edge_density_real": real_ed,
            "edge_density_gen":  gen_ed
        })

        print(f"[{idx+1}/{NUM_ENTRIES}] Processed: {file_path}")

    except Exception as e:
        # Any error (file missing, PIL error, NSFW, SSIM window‐size, etc.) skips this row
        print(f"[{idx+1}/{NUM_ENTRIES}] Skipped {row['file_name']} → {e}")
        continue        

# 10. Save all metrics into a CSV
metrics_df = pd.DataFrame(records)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"Saved metrics for {len(records)} examples (of {NUM_ENTRIES}) to {METRICS_CSV}")
