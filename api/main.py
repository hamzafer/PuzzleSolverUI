from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import base64
import torch

from .schemas import PuzzleRequest, PuzzleResponse, ModelInfo
from .models import get_available_models, solve_puzzle
from .utils import image_to_tensor, tensor_to_image, create_puzzle_grid

app = FastAPI(title="Puzzle Solver API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/models", response_model=list[ModelInfo])
async def list_models():
    """Get available puzzle solver models"""
    return get_available_models()

@app.post("/api/solve", response_model=PuzzleResponse)
async def solve(request: PuzzleRequest):
    """Solve a puzzle using the specified model"""
    try:
        # Convert base64 image to tensor
        puzzle_tensor = image_to_tensor(request.image_data)
        
        # Solve the puzzle
        solution_tensor, solution_info = solve_puzzle(puzzle_tensor, request.model_id)
        
        # Convert solution tensor to base64 image
        solution_image = tensor_to_image(solution_tensor)
        
        return PuzzleResponse(
            solution_image=solution_image,
            order=solution_info["order"],
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/create_puzzle")
async def create_puzzle(file: UploadFile = File(...)):
    """Create a random puzzle from an uploaded image"""
    try:
        # Read the uploaded file
        contents = await file.read()
        image_data = base64.b64encode(contents).decode('utf-8')
        
        # Convert to tensor
        image_tensor = image_to_tensor(image_data)
        
        # Create puzzle grid
        _, shuffled_tensor, indices = create_puzzle_grid(image_tensor)
        
        # Convert to base64
        shuffled_image = tensor_to_image(shuffled_tensor)
        
        return {
            "puzzle_image": shuffled_image,
            "indices": indices
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount the frontend
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html"), "r") as f:
        return f.read()