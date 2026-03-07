# API Documentation

The AI Face Detector API provides a REST interface for detecting AI-generated faces.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. For production use, implement API keys or OAuth.

## Endpoints

### 1. Root Endpoint

Get API information and available endpoints.

**Request:**
```http
GET /
```

**Response:**
```json
{
  "status": "success",
  "message": "AI Face Detector API",
  "version": "0.1.0",
  "endpoints": {
    "POST /detect": "Upload an image to detect if it's AI-generated or real",
    "GET /health": "Check API health status",
    "GET /": "API information"
  },
  "docs": "/docs"
}
```

### 2. Health Check

Check if the API is running and the model is loaded.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "device": "cuda",
  "version": "0.1.0"
}
```

**Status Codes:**
- `200 OK`: API is healthy
- `503 Service Unavailable`: Model not loaded

### 3. Detect Face

Upload an image and detect if it's AI-generated or real.

**Request:**
```http
POST /detect
Content-Type: multipart/form-data
```

**Parameters:**
- `file` (required): Image file (JPG, PNG, WEBP, max 5MB)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/image.jpg"
```

**Python Example:**
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )
    result = response.json()
    print(result)
```

**JavaScript Example:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(result => console.log(result));
```

**Response (Success):**
```json
{
  "status": "success",
  "result": "AI_GENERATED",
  "confidence": 0.98,
  "probabilities": {
    "AI_GENERATED": 0.98,
    "REAL": 0.02
  },
  "inference_time_ms": 45.2,
  "image_size": "512x512"
}
```

**Response Fields:**
- `status`: "success" or "error"
- `result`: "AI_GENERATED" or "REAL"
- `confidence`: Float (0-1), confidence score for the predicted class
- `probabilities`: Object with probabilities for both classes
- `inference_time_ms`: Float, time taken for inference in milliseconds
- `image_size`: String, dimensions of the uploaded image

**Status Codes:**
- `200 OK`: Successful inference
- `400 Bad Request`: Invalid file (wrong type, too large, etc.)
- `500 Internal Server Error`: Server error during inference
- `503 Service Unavailable`: Model not loaded

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

**Common Errors:**

| Status Code | Error | Description |
|-------------|-------|-------------|
| 400 | "Invalid file type. Please upload JPG, PNG, or WEBP image." | File extension not allowed |
| 400 | "File too large. Maximum size is 5MB." | File exceeds size limit |
| 400 | "File is empty." | Uploaded file has 0 bytes |
| 400 | "Failed to process image" | Image is corrupted or invalid |
| 503 | "Model not loaded. Please check server logs." | Model weights file missing |

## Rate Limiting

Currently, no rate limiting is implemented. For production, consider adding:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/detect")
@limiter.limit("10/minute")
async def detect_face(...):
    ...
```

## Interactive Documentation

When the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API documentation with test forms.

## Running the Server

### Development

```bash
# With auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# With multiple workers
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## CORS Configuration

By default, CORS is enabled for:
- `http://localhost`
- `http://localhost:8000`
- `http://localhost:3000`
- `http://127.0.0.1`
- `http://127.0.0.1:8000`

To add more origins, edit `app/config.py`:

```python
CORS_ORIGINS = [
    "https://yourdomain.com",
    "https://app.yourdomain.com"
]
```

## Testing the API

### Using the Frontend

1. Start the server: `uvicorn app.main:app --reload`
2. Open `static/index.html` in a browser
3. Upload an image to test

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Detect face
curl -X POST http://localhost:8000/detect \
  -F "file=@test_image.jpg"
```

### Using Python

```python
import requests

# Health check
response = requests.get('http://localhost:8000/health')
print(response.json())

# Detect face
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )
    print(response.json())
```

### Using Postman

1. Create a new POST request to `http://localhost:8000/detect`
2. Go to Body > form-data
3. Add key `file` with type `File`
4. Upload an image
5. Send request

## Performance Optimization

### Batch Inference

For multiple images, consider batch processing:

```python
@app.post("/detect-batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        # Process each file
        ...
    return results
```

### Caching

Cache frequent results:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def get_cached_prediction(image_hash: str):
    # Check cache before inference
    ...
```

### GPU Optimization

Ensure GPU is being used:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Monitoring

Add logging and monitoring:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    logger.info(f"Processing file: {file.filename}")
    # ... inference code
    logger.info(f"Result: {result}")
    return result
```

## Security Considerations

For production deployment:

1. **Authentication**: Add API keys or OAuth
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: Strict file type checking
4. **Size Limits**: Prevent DoS attacks
5. **HTTPS**: Use SSL/TLS
6. **Sanitization**: Clean file names and paths
7. **Logging**: Track usage and detect anomalies

## Support

For API issues or questions:
- Check the [GitHub Issues](https://github.com/furkankoykiran/ai-face-detector/issues)
- Review the error logs
- Test with the interactive documentation at `/docs`
