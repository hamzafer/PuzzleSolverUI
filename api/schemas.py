from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ModelInfo(BaseModel):
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")

class PuzzleRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    model_id: str = Field(..., description="The ID of the model to use")

class PuzzleResponse(BaseModel):
    solution_image: str = Field(..., description="Base64 encoded solution image")
    order: List[int] = Field(..., description="Ordering of the solution")
    success: bool = Field(True, description="Whether a solution was found")