import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Type
import matplotlib.pyplot as plt
from einops import rearrange
from sklearn.metrics import pairwise_distances
from PIL import Image
from torchvision import transforms

from .base_model import PuzzleSolverModel

# Add external module paths to Python path
sys.path.append('/cluster/home/muhamhz/fcvit-mt-ntnu')
sys.path.append('/cluster/home/muhamhz/JPDVT')
sys.path.append('/cluster/home/muhamhz/JPDVT/image_model')

# Import for the first model
from puzzle_fcvit import FCViT

# Imports for the second model
from diffusion import create_diffusion
from models import DiT_models, get_2d_sincos_pos_embed


class FCViTSolver(PuzzleSolverModel):
    """FCViT model for puzzle solving"""
    
    def __init__(self, checkpoint_path, backbone="vit_base_patch16_224", 
                 size_fragment=75, num_fragment=9, puzzle_size=225):
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.size_fragment = size_fragment
        self.num_fragment = num_fragment
        self.puzzle_size = puzzle_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @property
    def name(self) -> str:
        return "FCViT"
    
    @property
    def description(self) -> str:
        return "Vision Transformer model for solving image puzzles"
    
    def load_model(self) -> None:
        """Load FCViT model weights"""
        if self.model is not None:
            return
            
        self.model = FCViT(
            backbone=self.backbone, 
            num_fragment=self.num_fragment, 
            size_fragment=self.size_fragment
        ).to(self.device)
        
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.augment_fragment = transforms.Resize(
            (self.size_fragment, self.size_fragment), 
            antialias=True
        )
        
    def solve(self, puzzle_img: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Solve puzzle using FCViT model"""
        if self.model is None:
            self.load_model()
            
        # Move to device
        puzzle_img = puzzle_img.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            pred_gpu, tgt_gpu = self.model(puzzle_img)
        
        pred_ = self.model.mapping(pred_gpu.clone())
        map_coord = self.model.map_coord.cpu()
        
        # Extract the ordering information
        mask_pred = (pred_[0][:, None, :] == map_coord).all(-1).long()
        pred_indices = mask_pred.argmax(dim=1).tolist()
        
        # Visualize using model's unshuffling logic
        def unshuffle(tensor, order):
            C, H, W = tensor.shape
            p = self.size_fragment
            pieces = [tensor[:, i:i+p, j:j+p] for i in range(0, H, p) for j in range(0, W, p)]
            grid = [pieces[idx] for idx in order]
            rows = [torch.cat(grid[i:i+3], dim=2) for i in range(0, 9, 3)]
            return torch.cat(rows, dim=1)
        
        # Reconstruct the image
        img_cpu = puzzle_img[0].cpu()
        reconstructed = unshuffle(img_cpu, pred_indices)
        
        return reconstructed.unsqueeze(0), {"order": pred_indices}


class JPDVTSolver(PuzzleSolverModel):
    """JPDVT diffusion model for puzzle solving"""
    
    def __init__(self, checkpoint_path, model_name="JPDVT", image_size=192, grid_size=3, num_steps=250):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.model = None
        self.diffusion = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @property
    def name(self) -> str:
        return "JPDVT"
    
    @property
    def description(self) -> str:
        return "Diffusion-based model for solving image puzzles"
    
    def load_model(self) -> None:
        """Load JPDVT model weights"""
        if self.model is not None:
            return
            
        self.model = DiT_models[self.model_name](input_size=self.image_size).to(self.device)
        
        state_dict = torch.load(self.checkpoint_path, weights_only=False)
        model_state_dict = state_dict['model']
        
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in model_state_dict.items() if k in model_dict}
        self.model.load_state_dict(pretrained_dict, strict=False)
        
        # Because batchnorm doesn't work normally with batch size=1, set the model to train mode
        self.model.train()
        
        # Create the diffusion object
        self.diffusion = create_diffusion(str(self.num_steps))
        
    def find_permutation(self, distance_matrix):
        """
        Greedy algorithm to find the permutation order 
        based on the provided distance matrix.
        """
        sort_list = []
        for _ in range(distance_matrix.shape[1]):
            order = distance_matrix[:, 0].argmin()
            sort_list.append(order)
            distance_matrix = distance_matrix[:, 1:]
            distance_matrix[order, :] = 2024  # effectively removing that row
        return sort_list
        
    def solve(self, puzzle_img: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Solve puzzle using JPDVT model"""
        if self.model is None:
            self.load_model()
            
        # Move to device
        scrambled_img = puzzle_img.to(self.device)
        
        G = self.grid_size
        
        # Time embedding
        time_emb = torch.tensor(get_2d_sincos_pos_embed(8, G)).unsqueeze(0).float().to(self.device)
        time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, 12)).unsqueeze(0).float().to(self.device)
        time_emb_noise = torch.randn_like(time_emb_noise).repeat(1, 1, 1)
        
        # Split image into patches for visualization
        x_patches = rearrange(
            scrambled_img, 'b c (p1 h1) (p2 w1) -> b c (p1 p2) h1 w1', 
            p1=G, p2=G, h1=self.image_size//G, w1=self.image_size//G
        )
        
        # Store patches for reconstruction
        scrambled_patches = [x_patches[0, :, i, :, :] for i in range(G * G)]
        
        # Use the diffusion process to sample
        samples = self.diffusion.p_sample_loop(
            self.model.forward, 
            scrambled_img, 
            time_emb_noise.shape, 
            time_emb_noise, 
            clip_denoised=False, 
            model_kwargs=None, 
            progress=True, 
            device=self.device
        )
        
        # For the sample reordering, we use a downsampled patch size
        sample_patch_dim = self.image_size // (16 * G)
        
        # Rearrange to shape: (patches) x (some_size) x d
        sample = rearrange(samples[0], '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d', 
                          p1=G, p2=G, h1=sample_patch_dim, w1=sample_patch_dim)
        
        # Average across the spatial dimension to get a single feature per patch
        sample = sample.mean(1)
        
        # Compare with the time embedding
        dist = pairwise_distances(sample.cpu().numpy(), time_emb[0].cpu().numpy(), metric='manhattan')
        order = self.find_permutation(dist)
        pred = np.asarray(order).argsort()
        
        # Reconstruct the final puzzle using the predicted ordering
        reconstructed_patches = [None] * (G * G)
        for i, pos in enumerate(pred):
            reconstructed_patches[pos] = scrambled_patches[i]
            
        # Create a grid from the reconstructed patches
        reconstructed_img = torch.zeros_like(scrambled_img[0])
        
        # Reconstruct the full image by placing patches in the right positions
        patch_h = self.image_size // G
        patch_w = self.image_size // G
        
        for i, patch in enumerate(reconstructed_patches):
            row = i // G
            col = i % G
            reconstructed_img[:, row*patch_h:(row+1)*patch_h, col*patch_w:(col+1)*patch_w] = patch
            
        return reconstructed_img.unsqueeze(0), {"order": pred.tolist()}


# Registry of available models
def get_available_models() -> List[Dict[str, str]]:
    """Get list of available models"""
    models = [
        {
            "id": "fcvit",
            "name": "FCViT",
            "description": "Vision Transformer model for solving image puzzles"
        },
        {
            "id": "jpdvt",
            "name": "JPDVT",
            "description": "Diffusion-based model for solving image puzzles"
        }
    ]
    return models

def get_model_instance(model_id: str) -> PuzzleSolverModel:
    """Get model instance by ID"""
    if model_id == "fcvit":
        ckpt_path = os.path.join(os.getcwd(), "checkpoints/FCViT_base_3x3_ep100_lr3e-05_b64.pt")
        return FCViTSolver(checkpoint_path=ckpt_path)
    elif model_id == "jpdvt":
        ckpt_path = os.path.join(os.getcwd(), "checkpoints/2850000.pt")
        return JPDVTSolver(checkpoint_path=ckpt_path)
    else:
        raise ValueError(f"Model '{model_id}' not found")

def solve_puzzle(puzzle_img: torch.Tensor, model_id: str) -> Tuple[torch.Tensor, Dict]:
    """Solve puzzle using specified model"""
    model = get_model_instance(model_id)
    return model.solve(puzzle_img)