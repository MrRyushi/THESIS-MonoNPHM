import numpy as np
import torch
from PIL import Image
from tracking import render_image


# Load the saved transformation parameters
rot = np.load("../../../tracking_output/pretrained_mononphm_original/stage2/510_seq_4/00000/rot.npy")  # Rotation matrix
trans = np.load("../../../tracking_output/pretrained_mononphm_original/stage2/510_seq_4/00000/trans.npy")  # Translation
scale = np.load("../../../tracking_output/pretrained_mononphm_original/stage2/510_seq_4/00000/scale.npy")  # Scaling factor
z_exp = np.load("../../../tracking_output/pretrained_mononphm_original/stage2/510_seq_4/00000/z_exp.npy")  # Expression latent code

# Define 90-degree rotation around the Y-axis
Ry_90 = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
])

# Apply the 90-degree rotation to the existing rotation matrix
rot_90 = rot @ Ry_90  # Matrix multiplication to apply transformation

# Convert to Torch tensors for rendering
rot_tensor = torch.from_numpy(rot_90).float().cuda()
trans_tensor = torch.from_numpy(trans).float().cuda()
scale_tensor = torch.from_numpy(scale).float().cuda()

# Prepare pose parameters
pose_params = [rot_tensor, trans_tensor, scale_tensor]

# Define rendering condition
condition = {
    "geo": torch.zeros((1, 1, 256)).cuda(),  # Assuming latent space is 256D, modify accordingly
    "app": torch.zeros((1, 1, 256)).cuda(),
    "exp": torch.from_numpy(z_exp).unsqueeze(0).float().cuda()
}

# Call the existing render_image function
I, I_o_mask, I_pred_o_mask, sdf, weights_sum, weights_img, depths = render_image(
    idr,  # This should be your VolumetricRenderer instance
    expression=0,  # Assuming single expression
    condition=condition,
    sh_coeffs=None,  # Adjust if using spherical harmonics lighting
    in_dict=input_data,  # Pass input dictionary from dataset
    w=512, h=512,  # Set output resolution
    scale_uniformly=True,
    pose_params=pose_params,
    use_sh=True
)

# Save the rendered image
I.save("rendered_90_degree.png")
print("Saved 90-degree render as 'rendered_90_degree.png'")
