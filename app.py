from fastapi import FastAPI, UploadFile, File, HTTPException,Query
from inference import predict_image
from PIL import Image
import io
app = FastAPI(
    title="Bengali Handwritten Subdistrict-District Pair Name Recognition",
    description="Predict the name of subdistrict and its respective district from a handwritten image",
    version="1.0.0"
)
@app.get("/")
def root():
    return {
        "message": "Bengali Subdistrict-District Classifier API is running!"
    }

@app.post("/predict")
async def predict(version: str = Query("v1"),file: UploadFile = File(...)):
    
    if version not in ["v1", "v2"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid model version."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    contents = await file.read()
   
    image = Image.open(io.BytesIO(contents))
        
    result = predict_image(image,version)
       
    return result

    

    