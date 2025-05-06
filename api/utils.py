import base64
import io
from typing import Tuple
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Transform for preprocessing images
def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

def image_to_tensor(image_data: str, image_size=224) -> torch.Tensor:
    """Convert base64 image to tensor"""
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = get_transform(image_size)
    return transform(image).unsqueeze(0)  # Add batch dimension

def tensor_to_image(tensor: torch.Tensor) -> str:
    """Convert tensor to base64 image"""
    # Ensure the tensor is on CPU and detached
    tensor = tensor.cpu().detach()
    
    # If batch dimension is present, take the first item
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Convert to PIL image
    tensor = tensor.clamp(0, 1) # Ensure values are in [0, 1]
    image = transforms.ToPILImage()(tensor)
    
    # Save to bytes and convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    
    return base64.b64encode(image_bytes).decode('utf-8')

def create_puzzle_grid(image: torch.Tensor, grid_size=3) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Create a puzzle by randomly reordering the grid cells of an image
    
    Returns:
        Tuple containing:
        - Original image tensor
        - Shuffled puzzle tensor
        - Indices used for shuffling
    """
    # Ensure the tensor is on CPU
    image = image.cpu()
    
    # If batch dimension is present, take the first item
    if image.dim() == 4:
        image = image[0]
    
    c, h, w = image.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    
    # Split the image into grid cells
    cells = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = image[:, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cells.append(cell)
    
    # Randomly shuffle the cells
    indices = np.random.permutation(grid_size * grid_size)
    shuffled_cells = [cells[i] for i in indices]
    
    # Reconstruct the shuffled image
    shuffled = torch.zeros_like(image)
    for idx, cell in enumerate(shuffled_cells):
        i, j = idx // grid_size, idx % grid_size
        shuffled[:, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = cell
    
    return image.unsqueeze(0), shuffled.unsqueeze(0), indices.tolist()