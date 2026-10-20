import base64
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fastapi.middleware.cors import CORSMiddleware
from .schemas import HealthResponse, PredictResponse
from fastapi import FastAPI, File, HTTPException, UploadFile

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_WIDTH = 800
MAX_HEIGHT = 600

app = FastAPI(
    title="Bubbles Segmentation API",
    description="Backend API for bubbles segmentation web application.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:4201",
    "https://bubbles.olver.site",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("models/best.pt")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Bubbles Segmentation API"}

@app.get(
    "/health",
    summary="Check API health",
    description="Checks whether the API is running and the required model is available.",
    tags=["Health"],
    response_model=HealthResponse
)
def health():
    check_files = [
        "runs/semantic/experiment_cloud_nano/weights/best.pt"
    ]
    files_ok = all(Path(file).exists() for file in check_files)
    return {
        "status": "ok" if files_ok else "error",
        "files_available": files_ok,
    }

@app.post("/predict-bubble-segmentation",
          summary="Predict Bubble Segmentation",
          description="Perform bubble segmentation and calculate the bubble area percentage.",
          tags=["Predict Segmentation"],
          response_model=PredictResponse
          )
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image must not exceed 5 MB.")
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported image format")

    try:
        pil_image = Image.open(image.file).convert("RGB")
        if pil_image.size != (MAX_WIDTH, MAX_HEIGHT):
            raise HTTPException(status_code=400, detail="Image dimensions must be exactly 800x600 pixels")

        results = model.predict(
            source=pil_image,
            imgsz=640,
            verbose=False,
        )

        mask = results[0].semantic_mask.data.cpu().numpy()

        mask_bubble_pixels = np.count_nonzero(mask == 1)
        mask_bubble_area_percentage = mask_bubble_pixels / mask.size * 100
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
        cmap = ListedColormap(["blue", "red"])

        fig.patch.set_alpha(0)
        for a in ax:
            a.set_facecolor("none")

        ax[0].imshow(pil_image)
        ax[0].set_title("Original", color="white", fontsize=20)
        ax[0].axis("off")

        ax[1].imshow(pil_image)
        ax[1].imshow(mask, alpha=0.5, cmap=cmap, vmin=0, vmax=1)
        ax[1].contour(mask, levels=[0.5], colors=["red"], linewidths=1)
        ax[1].set_title(f"Prediction - BAP: {mask_bubble_area_percentage:.5f}", color="white", fontsize=20)
        ax[1].axis("off")
        
        fig.tight_layout()

        buffer = BytesIO()
        fig.savefig(
            buffer, 
            format="png", 
            bbox_inches="tight", 
            dpi=100,
            transparent=True
            )
        plt.close(fig)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "width": int(pil_image.width),
            "height": int(pil_image.height),
            "image": image_base64,
            "bubble_area_percentage": float(mask_bubble_area_percentage),
        }


    except Exception as exc:
        print(exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to process image",
        ) from exc

def main():
    uvicorn.run("bubbles.main:app", host="0.0.0.0", port=8000)