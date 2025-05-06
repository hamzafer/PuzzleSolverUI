from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
import base64
import torch
import logging
from typing import List  # Add this import

from .schemas import PuzzleRequest, PuzzleResponse, ModelInfo
from .models import get_available_models, solve_puzzle
from .utils import image_to_tensor, tensor_to_image, create_puzzle_grid

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Puzzle Solver API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/models", response_model=List[ModelInfo])  # Change this line
async def list_models():
    """Get available puzzle solver models"""
    return get_available_models()

@app.post("/api/solve", response_model=PuzzleResponse)
async def solve(request: PuzzleRequest):
    """Solve a puzzle using the specified model"""
    try:
        # Log the request parameters (excluding the actual image data for brevity)
        logger.debug(f"Solving puzzle with model_id: {request.model_id}")
        
        # Different image sizes for different models
        image_size = 225 if request.model_id == "fcvit" else 192
        
        # Convert base64 image to tensor
        try:
            puzzle_tensor = image_to_tensor(request.image_data, image_size)
            logger.debug(f"Image tensor shape: {puzzle_tensor.shape}")
        except Exception as e:
            logger.error(f"Error converting image to tensor: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
        
        # Solve the puzzle
        try:
            logger.debug(f"Starting to solve with model: {request.model_id}")
            solution_tensor, solution_info = solve_puzzle(puzzle_tensor, request.model_id)
            logger.debug(f"Solution found with info: {solution_info}")
        except Exception as e:
            logger.error(f"Error during puzzle solving: {str(e)}", exc_info=True)
            raise ValueError(f"Puzzle solving failed: {str(e)}")
        
        # Convert solution tensor to base64 image
        try:
            solution_image = tensor_to_image(solution_tensor)
        except Exception as e:
            logger.error(f"Error converting solution to image: {str(e)}")
            raise ValueError(f"Failed to create solution image: {str(e)}")
        
        return PuzzleResponse(
            solution_image=solution_image,
            order=solution_info["order"],
            success=True
        )
    except Exception as e:
        logger.error(f"Error in solve endpoint: {str(e)}", exc_info=True)
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