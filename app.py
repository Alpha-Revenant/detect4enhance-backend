from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf

# Load the model
MODEL_PATH = "engagement_model_89.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Labels
labels = ["Engaged", "Frustrated", "Bored", "Confused"]

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Engagement Detection API',
        'endpoints': {
            '/predict': 'POST image for emotion prediction'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    predictions = {"Engaged": 0, "Frustrated": 0, "Bored": 0, "Confused": 0}

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            img_resized = cv2.resize(face, (224, 224))
            img_resized = img_resized.astype("float32") / 255.0
            img_resized = np.expand_dims(img_resized, axis=0)

            # Get model predictions
            preds = predict_tflite(img_resized)[0]
            
            # Normalize to sum to 1 (100%)
            preds = preds / np.sum(preds)
            
            # Get dominant emotion
            dominant_idx = np.argmax(preds)
            dominant_emotion = labels[dominant_idx]
            
            # Debug output
            print(f"Model predictions: {dict(zip(labels, preds))}")
            print(f"Dominant emotion: {dominant_emotion}")
            
            # Store results
            for i, label in enumerate(labels):
                predictions[label] = float(preds[i])
            result = dominant_emotion

    if len(faces) == 0:
        return jsonify({"error": "No face detected", "predictions": None, "result": None})
    return jsonify({"predictions": predictions, "result": dominant_emotion})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host='0.0.0.0', port=port, debug=True)
