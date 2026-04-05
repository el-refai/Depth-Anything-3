import glob, os, torch
import numpy as np
from PIL import Image
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.alignment import apply_metric_scaling
device = torch.device("cuda")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device=device)
example_path = "assets/exampsles/SOH"
breakpoint()
# images = sorted(glob.glob(os.path.join(example_path, "*.png")))
# prediction = model.inference(
#     images,
# )

# images = ["/home/karimelrafi/sam-3d-objects/custom_data/color_l_0.png"]
# prediction = model.inference(images, process_res=1920)

# images = ["/home/karimelrafi/sam-3d-objects/custom_data/omni-scan/Screenshot.png"]
images = ["/home/karimelrafi/Depth-Anything-3/output_depth/pasted-movie.png"]
prediction = model.inference(images)


# Create output directory
output_dir = "output_depth"
os.makedirs(output_dir, exist_ok=True)

# Save depth predictions
depth = prediction.depth

metric_depth = apply_metric_scaling(depth, prediction.intrinsics, prediction.scale_factor)
if isinstance(depth, torch.Tensor):
    depth = depth.cpu().numpy()

# Handle batch dimension
if len(depth.shape) == 4:  # [B, C, H, W]
    depth = depth.squeeze(1)  # Remove channel dimension -> [B, H, W]

if len(metric_depth.shape) == 4:  # [B, C, H, W]
    metric_depth = metric_depth.squeeze(1)  # Remove channel dimension -> [B, H, W]

# Save each depth map
for idx, img_path in enumerate(images):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Save as NPY
    npy_path = os.path.join(output_dir, f"{base_name}_depth.npy")
    np.save(npy_path, depth[idx])

    metric_npy_path = os.path.join(output_dir, f"{base_name}_metric_depth.npy")
    np.save(metric_npy_path, metric_depth[idx])
    print(f"Saved: {npy_path}")
    
    # Normalize depth for PNG visualization (0-255)
    depth_normalized = (depth[idx] - depth[idx].min()) / (depth[idx].max() - depth[idx].min())
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Save as PNG
    png_path = os.path.join(output_dir, f"{base_name}_depth.png")
    Image.fromarray(depth_uint8).save(png_path)
    print(f"Saved: {png_path}")
