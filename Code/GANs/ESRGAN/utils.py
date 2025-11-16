import torch
import os
import config
import numpy as np
from PIL import Image
from torchvision.utils import save_image


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(f"=> Looking for checkpoint at: {checkpoint_file}")
    if not os.path.exists(checkpoint_file):
        print(f"=> Checkpoint file {checkpoint_file} not found. Starting with random weights.")
        return False
    
    print(f"=> Loading checkpoint from: {checkpoint_file}")
    print(f"=> File size: {os.path.getsize(checkpoint_file)} bytes")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        print(f"=> Checkpoint type: {type(checkpoint)}")
        
        # Debug: show checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"=> Checkpoint keys: {list(checkpoint.keys())}")
            if "state_dict" in checkpoint:
                print(f"=> State dict keys (first 5): {list(checkpoint['state_dict'].keys())[:5]}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                # Standard format with state_dict and optimizer
                print("=> Loading from 'state_dict' key...")
                model.load_state_dict(checkpoint["state_dict"])
                if "optimizer" in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    # Update learning rate
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
            else:
                # Direct state dict format (keys are layer names)
                print("=> Loading as direct state dict...")
                print(f"=> First 5 keys: {list(checkpoint.keys())[:5]}")
                model.load_state_dict(checkpoint)
                print("=> Loaded model weights only (no optimizer state)")
        else:
            # Assume it's a direct state dict
            print("=> Loading non-dict checkpoint...")
            model.load_state_dict(checkpoint)
            print("=> Loaded model weights only (no optimizer state)")
            
        print("=> Checkpoint loaded successfully")
        return True
    except Exception as e:
        print(f"=> Error loading checkpoint: {e}")
        print(f"=> Error type: {type(e)}")
        import traceback
        print(f"=> Full traceback: {traceback.format_exc()}")
        print("=> Continuing with random weights...")
        return False


def plot_examples(low_res_folder, gen):
    # Create absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_images_path = os.path.join(script_dir, low_res_folder)
    saved_path = os.path.join(script_dir, "saved")
    
    # Create saved directory if it doesn't exist
    os.makedirs(saved_path, exist_ok=True)
    
    if not os.path.exists(test_images_path):
        print(f"=> Test images directory {test_images_path} not found.")
        return
        
    files = [f for f in os.listdir(test_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not files:
        print(f"=> No image files found in {test_images_path}")
        return

    gen.eval()
    print(f"=> Processing {len(files)} images...")
    for file in files:
        try:
            image_path = os.path.join(test_images_path, file)
            image = Image.open(image_path)
            with torch.no_grad():
                # Convert image to tensor
                input_tensor = config.test_transform(image=np.asarray(image))["image"]
                input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)
                
                # Generate upscaled image
                upscaled_img = gen(input_tensor)
                
                # Clamp values to valid range and normalize for saving
                upscaled_img = torch.clamp(upscaled_img, -1, 1)  # Clamp to [-1, 1]
                upscaled_img = (upscaled_img + 1) / 2  # Normalize to [0, 1]
                
            output_path = os.path.join(saved_path, file)
            save_image(upscaled_img, output_path)
            print(f"=> Input shape: {input_tensor.shape}, Output shape: {upscaled_img.shape}")
            print(f"=> Saved upscaled image: {output_path}")
        except Exception as e:
            print(f"=> Error processing {file}: {e}")
    gen.train()
