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

# Import for the first model FCViT
from puzzle_fcvit import FCViT

# Imports for the second model JPDVT
from diffusion import create_diffusion
import models as jpdvt_models            # JPDVT/image_model/models.py
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
        
        # Make sure map_coord is on the same device as pred_
        map_coord = self.model.map_coord.to(pred_.device)
        
        # Extract the ordering information
        mask_pred = (pred_[0][:, None, :] == map_coord).all(-1).long()
        pred_indices = mask_pred.argmax(dim=1).tolist()
        
        # Move tensors back to CPU for post-processing
        img_cpu = puzzle_img[0].cpu()
        
        # Visualize using model's unshuffling logic
        def unshuffle(tensor, order):
            C, H, W = tensor.shape
            p = self.size_fragment
            pieces = [tensor[:, i:i+p, j:j+p] for i in range(0, H, p) for j in range(0, W, p)]
            grid = [pieces[idx] for idx in order]
            rows = [torch.cat(grid[i:i+3], dim=2) for i in range(0, 9, 3)]
            return torch.cat(rows, dim=1)
        
        # Reconstruct the image
        reconstructed = unshuffle(img_cpu, pred_indices)
        
        return reconstructed.unsqueeze(0), {"order": pred_indices}


class JPDVTSolver(PuzzleSolverModel):
    """JPDVT diffusion model for puzzle solving (fully working port)"""
    
    def __init__(self, checkpoint_path, model_name="JPDVT", image_size=192, grid_size=3, num_steps=250):
        self.checkpoint_path = checkpoint_path
        self.model_name      = model_name
        self.image_size      = image_size
        self.grid_size       = grid_size
        self.num_steps       = num_steps
        self.model           = None
        self.diffusion       = None
        self.device          = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def name(self) -> str:
        return "JPDVT"
    
    @property
    def description(self) -> str:
        return "Diffusion-based model for solving image puzzles"
    
    def load_model(self) -> None:
        """Load JPDVT model weights and patch the embedder"""
        if self.model is not None:
            return
            
        # Monkey-patch PatchEmbed to swallow bias kwarg
        if not hasattr(jpdvt_models.PatchEmbed, "_orig_init"):
            jpdvt_models.PatchEmbed._orig_init = jpdvt_models.PatchEmbed.__init__
            def _patched_init(self, img_size, patch_size, in_chans, embed_dim, bias=False, *a, **k):
                return jpdvt_models.PatchEmbed._orig_init(
                    self, img_size, patch_size, in_chans, embed_dim, *a, **k
                )
            jpdvt_models.PatchEmbed.__init__ = _patched_init

        # Instantiate the DiT model
        self.model = DiT_models[self.model_name](input_size=self.image_size).to(self.device)
        
        # Load checkpoint (handles both {"model":{...}} or raw)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        sd = state_dict.get('model', state_dict)
        own = self.model.state_dict()
        pretrained = {k: v for k, v in sd.items() if k in own}
        self.model.load_state_dict(pretrained, strict=False)
        
        # BatchNorm hack
        self.model.train()
        
        # Create diffusion sampler
        self.diffusion = create_diffusion(str(self.num_steps))
        
    def find_permutation(self, distance_matrix: np.ndarray) -> List[int]:
        """Greedy algorithm, safe on empty arrays."""
        sort_list = []
        tmp = distance_matrix.copy()
        n = distance_matrix.shape[1]
        for _ in range(n):
            idx = int(tmp[:, 0].argmin())
            sort_list.append(idx)
            tmp = tmp[:, 1:]
            if tmp.size > 0:
                tmp[idx, :] = tmp.max() + 1
        return sort_list
        
    def solve(self, puzzle_img: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Solve puzzle using the exact sequence of your standalone script."""
        if self.model is None:
            self.load_model()
            
        # 1) Prepare
        x = puzzle_img.to(self.device) * 2.0 - 1.0   # [1,3,H,W] in [-1,1]
        G = self.grid_size
        IMG_SZ = self.image_size
        
        # 2) Time embeddings
        time_emb       = torch.tensor(get_2d_sincos_pos_embed(8, G))\
                              .unsqueeze(0).float().to(self.device)     # [1,9,8]
        tokens_side    = IMG_SZ // 16
        noise_emb      = torch.tensor(get_2d_sincos_pos_embed(8, tokens_side))\
                              .unsqueeze(0).float().to(self.device)     # [1,144,8]
        noise          = torch.randn_like(noise_emb)                     # [1,144,8]
        
        # 3) Diffusion in 8-D latent
        samples = self.diffusion.p_sample_loop(
            self.model.forward,
            x,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=None,
            device=self.device,
            progress=False
        )
        latent = samples[0]  # [144,8]
        
        # 4) Group into 3×3 patches (4×4 tokens each)
        tp = tokens_side // G
        feat = rearrange(
            latent,
            '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d',
            p1=G, p2=G, h1=tp, w1=tp
        )  # [9,16,8]
        feat = feat.mean(1).cpu().numpy()  # [9,8]
        
        # 5) Greedy match
        target = time_emb[0].cpu().numpy()  # [9,8]
        dist   = pairwise_distances(feat, target, metric='manhattan')
        order  = self.find_permutation(dist)
        inv    = np.argsort(order).tolist()
        
        # 6) Reassemble pixels
        pix = rearrange(
            x[0],
            'c (g1 h) (g2 w) -> (g1 g2) c h w',
            g1=G, g2=G,
            h=IMG_SZ//G, w=IMG_SZ//G
        )  # [9,3,H/3,W/3]
        rec_patches = [pix[i] for i in inv]
        rec = rearrange(
            torch.stack(rec_patches),
            '(g1 g2) c h w -> c (g1 h) (g2 w)',
            g1=G, g2=G,
            h=IMG_SZ//G, w=IMG_SZ//G
        )  # [3,H,W]
        
        # 7) Back to [0,1]
        rec = rec * 0.5 + 0.5
        return rec.unsqueeze(0), {"order": inv}


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
    
    # Resize the image if needed
    if model_id == "fcvit" and puzzle_img.shape[-1] != 225:
        # Resize to 225x225 for FCViT
        transform = transforms.Resize((225, 225), antialias=True)
        puzzle_img = transform(puzzle_img)
    elif model_id == "jpdvt" and puzzle_img.shape[-1] != 192:
        # Resize to 192x192 for JPDVT
        transform = transforms.Resize((192, 192), antialias=True)
        puzzle_img = transform(puzzle_img)
    
    return model.solve(puzzle_img)
