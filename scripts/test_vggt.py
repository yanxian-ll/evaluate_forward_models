import torch

from mapanything.utils.image import load_images
from mapanything.models import init_model_from_config


# Set the cache directory for torch hub
torch.hub.set_dir("/opt/data/private/code/map-anything/checkpoints/torch_cache/hub")

# load local dino repo
LOCAL_DINO_REPO = "/opt/data/private/code/map-anything/checkpoints/torch_cache/hub/facebookresearch_dinov2_main"
_original_torch_hub_load = torch.hub.load
def offline_torch_hub_load(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == "facebookresearch/dinov2":
        print("Redirecting DINOv2 torch.hub.load to local repo")
        repo_or_dir = LOCAL_DINO_REPO
        kwargs["source"] = "local"
    return _original_torch_hub_load(repo_or_dir, model, *args, **kwargs)
torch.hub.load = offline_torch_hub_load


model = init_model_from_config("vggt", device="cuda")

# # Load and preprocess images from a folder
# # This handles resizing and normalization based on model requirements
# views = load_images(
#     folder_or_list="path/to/images",  # Folder path or list of image paths
#     resolution_set=518,               # Model-specific resolution (see table above)
#     norm_type="dinov2",               # Model-specific normalization (see table above)
#     patch_size=14,
# )

# # Run inference
# model.eval()
# with torch.no_grad():
#     with torch.autocast("cuda"):
#         predictions = model(views)

