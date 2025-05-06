# api/models.py

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from einops import rearrange
from sklearn.metrics import pairwise_distances

# Make sure we can import JPDVT internals
sys.path.append('/cluster/home/muhamhz/JPDVT/image_model')
from diffusion import create_diffusion
import models as jpdvt_models            # JPDVT/image_model/models.py
from models import DiT_models, get_2d_sincos_pos_embed

# Make sure we can import FCViT
sys.path.append('/cluster/home/muhamhz/fcvit-mt-ntnu')
from puzzle_fcvit import FCViT
from torchvision import transforms

# Your base interface
from .base_model import PuzzleSolverModel


class JPDVTSolver(PuzzleSolverModel):
    """Exact port of your working JPDVT inference for a single scrambled image."""

    def __init__(self,
                 checkpoint_path: str,
                 image_size: int = 192,
                 grid_size: int = 3,
                 num_steps: int = 250):
        self.checkpoint_path = checkpoint_path
        self.image_size      = image_size
        self.grid_size       = grid_size
        self.num_steps       = num_steps
        self.device          = "cuda" if torch.cuda.is_available() else "cpu"
        self.model           = None
        self.diffusion       = None

    @property
    def name(self) -> str:
        return "jpdvt"

    @property
    def description(self) -> str:
        return "Diffusion-based JPDVT solver"

    def load_model(self) -> None:
        if self.model is not None:
            return

        # Monkey-patch PatchEmbed to ignore bias kwarg
        if not hasattr(jpdvt_models.PatchEmbed, "_orig_init"):
            jpdvt_models.PatchEmbed._orig_init = jpdvt_models.PatchEmbed.__init__
            def _patched_init(self, img_size, patch_size, in_chans, embed_dim, bias=False, *a, **k):
                return jpdvt_models.PatchEmbed._orig_init(
                    self, img_size, patch_size, in_chans, embed_dim, *a, **k
                )
            jpdvt_models.PatchEmbed.__init__ = _patched_init

        # 1) instantiate DiT
        self.model = DiT_models["JPDVT"](input_size=self.image_size).to(self.device)

        # 2) load checkpoint
        state = torch.load(self.checkpoint_path, map_location=self.device)
        sd = state.get("model", state)
        own = self.model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in own}
        self.model.load_state_dict(filtered, strict=False)

        # 3) batchnorm mode
        self.model.train()

        # 4) diffusion sampler
        self.diffusion = create_diffusion(str(self.num_steps))

    def find_permutation(self, dist_mat: np.ndarray) -> List[int]:
        """
        Greedy algorithm to find the permutation order based on the distance matrix.
        Handles empty arrays gracefully by skipping assignment when empty.
        """
        sort_list = []
        tmp = dist_mat.copy()
        n = dist_mat.shape[1]
        for _ in range(n):
            # pick the patch with smallest distance in the first column
            idx = int(tmp[:, 0].argmin())
            sort_list.append(idx)
            # remove that first column
            tmp = tmp[:, 1:]
            # if there's still data, mark this row so it's not picked again
            if tmp.size > 0:
                tmp[idx, :] = tmp.max() + 1
        return sort_list


    def solve(self, puzzle_img: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """
        Args:
          puzzle_img: Tensor [1,3,H,W] in [0,1], already scrambled
        Returns:
          reconstructed [1,3,H,W], and {"order": [...]}
        """
        self.load_model()
        DEVICE = self.device
        IMG_SZ = self.image_size
        G      = self.grid_size

        # 1) Convert input to [-1,1] as in your script
        x = puzzle_img.to(DEVICE) * 2.0 - 1.0  # [1,3,H,W]

        # 2) Build the 8-D time embeddings
        #    a) per-puzzle-piece target (3x3 → 9 vectors of dim 8)
        time_emb = torch.tensor(
            get_2d_sincos_pos_embed(8, G)
        ).unsqueeze(0).float().to(DEVICE)       # [1,9,8]
        #    b) per-token noise (12x12 → 144 vectors of dim 8)
        tokens_per_side = IMG_SZ // 16
        noise_emb = torch.tensor(
            get_2d_sincos_pos_embed(8, tokens_per_side)
        ).unsqueeze(0).float().to(DEVICE)       # [1,144,8]
        noise = torch.randn_like(noise_emb)    # [1,144,8]

        # 3) Run diffusion in that 8-D latent space
        samples = self.diffusion.p_sample_loop(
            self.model.forward,
            x,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=None,
            device=DEVICE,
            progress=False
        )
        # samples: [1,144,8]
        latent = samples[0]  # [144,8]

        # 4) Group the 144 token-vectors into 3x3=9 pieces, each 4x4 tokens
        tp = tokens_per_side // G  # e.g. 12//3 = 4
        feat = rearrange(
            latent,
            '(p1 h1 p2 w1) d -> (p1 p2) (h1 w1) d',
            p1=G, p2=G, h1=tp, w1=tp
        )  # [9,16,8]
        feat = feat.mean(1).cpu().numpy()  # [9,8]

        # 5) Compare to the 3x3 positional embeddings
        target = time_emb[0].cpu().numpy()  # [9,8]
        dist   = pairwise_distances(feat, target, metric="manhattan")
        order  = self.find_permutation(dist)      # list of length 9
        inv    = np.argsort(order).tolist()      # inverse permutation

        # 6) Reassemble the scrambled image on pixel side
        pix = rearrange(
            x[0],
            'c (g1 h) (g2 w) -> (g1 g2) c h w',
            g1=G, g2=G,
            h=IMG_SZ//G,
            w=IMG_SZ//G
        )  # [9,3,H/3,W/3]
        rec_patches = [pix[i] for i in inv]
        rec = rearrange(
            torch.stack(rec_patches),
            '(g1 g2) c h w -> c (g1 h) (g2 w)',
            g1=G, g2=G,
            h=IMG_SZ//G,
            w=IMG_SZ//G
        )  # [3,H,W]

        # 7) back to [0,1]
        rec = rec * 0.5 + 0.5
        return rec.unsqueeze(0), {"order": inv}


class FCViTSolver(PuzzleSolverModel):
    """Vision‐Transformer–based solver (FCViT)"""

    def __init__(self,
                 checkpoint_path: str,
                 backbone: str = "vit_base_patch16_224",
                 num_fragment: int = 9,
                 frag_size: int = 75,
                 puzzle_size: int = 225):
        self.checkpoint_path = checkpoint_path
        self.backbone        = backbone
        self.num_fragment    = num_fragment
        self.frag_size       = frag_size
        self.puzzle_size     = puzzle_size
        self.device          = "cuda" if torch.cuda.is_available() else "cpu"
        self.model           = None

    @property
    def name(self) -> str:
        return "fcvit"

    @property
    def description(self) -> str:
        return "Vision Transformer solver (FCViT)"

    def load_model(self) -> None:
        if self.model is not None:
            return
        ckpt   = torch.load(self.checkpoint_path, map_location="cpu")["model"]
        state  = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        self.model = FCViT(
            backbone=self.backbone,
            num_fragment=self.num_fragment,
            size_fragment=self.frag_size
        ).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model.augment_fragment = transforms.Resize(
            (self.frag_size, self.frag_size), antialias=True
        )

    def solve(self, puzzle_img: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        self.load_model()
        img = puzzle_img.to(self.device)
        with torch.no_grad():
            pred, tgt = self.model(img)

        pred_map = self.model.mapping(pred.clone()).cpu()
        coord    = self.model.map_coord.cpu()
        mask     = (pred_map[:, None, :] == coord).all(-1).long()
        idx      = mask.argmax(-1).tolist()

        # rebuild pixel grid
        def unshuffle(t: torch.Tensor, order: List[int]):
            C, H, W = t.shape
            p        = self.frag_size
            pieces   = [
                t[:, i:i+p, j:j+p]
                for i in range(0, H, p)
                for j in range(0, W, p)
            ]
            side = int(np.sqrt(len(order)))
            grid = [pieces[o] for o in order]
            rows = [
                torch.cat(grid[i:i+side], dim=2)
                for i in range(0, len(order), side)
            ]
            return torch.cat(rows, dim=1)

        rec = unshuffle(img[0].cpu(), idx)
        rec = (rec - rec.min()) / (rec.max() - rec.min())
        return rec.unsqueeze(0), {"order": idx}


# -------------------------------------------------------------------------------
# Registry for FastAPI
# -------------------------------------------------------------------------------
def get_available_models() -> List[Dict[str, str]]:
    return [
        {"id": m.name, "name": m.name.upper(), "description": m.description}
        for m in [
            JPDVTSolver(checkpoint_path=os.path.join("checkpoints", "2850000.pt")),
            FCViTSolver(checkpoint_path=os.path.join("checkpoints", "FCViT_base_3x3_ep100_lr3e-05_b64.pt"))
        ]
    ]


def get_model_instance(model_id: str) -> PuzzleSolverModel:
    if model_id == "jpdvt":
        return JPDVTSolver(checkpoint_path=os.path.join("checkpoints", "2850000.pt"))
    if model_id == "fcvit":
        return FCViTSolver(checkpoint_path=os.path.join("checkpoints", "FCViT_base_3x3_ep100_lr3e-05_b64.pt"))
    raise ValueError(f"Unknown model_id: {model_id}")


def solve_puzzle(puzzle_img: torch.Tensor, model_id: str) -> Tuple[torch.Tensor, Any]:
    solver = get_model_instance(model_id)
    return solver.solve(puzzle_img)
