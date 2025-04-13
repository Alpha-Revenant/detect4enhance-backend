# Engagement Detection API

A Flask-based API for detecting engagement levels from facial expressions using TensorFlow Lite.

## API Endpoints

- `GET /`: Returns API information
- `POST /predict`: Accepts an image file and returns engagement predictions

## Deployment

1. Clone this repository
2. Build Docker image: `docker build -t engagement-api .`
3. Run container: `docker run -p 7860:7860 engagement-api`

## Example Usage

```python
import requests

url = "https://your-huggingface-space-url.hf.space/predict"
files = {'image': open('test.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

## Model Details

- Model: Custom TensorFlow Lite model (`engagement_model_89.tflite`)
- Input: 224x224 RGB face image
- Output: Probabilities for 4 engagement states:
  - Engaged
  - Frustrated
  - Bored
  - Confused
