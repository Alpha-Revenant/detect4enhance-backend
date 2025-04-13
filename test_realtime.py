import cv2
import requests

# URL of your local Flask server
URL = "http://127.0.0.1:5000/predict"

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

    try:
        # Send to Flask API
        response = requests.post(URL, files=files)
        result = response.json()

        predictions = result.get('predictions', {})
        result_text = result.get('result', 'Unknown')

        # Display probabilities for each label
        y_offset = 30
        for label, prob in predictions.items():
            text = f"{label}: {prob:.2f}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

        # Display the final result on top
        cv2.putText(frame, f"Final Result: {result_text}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    except Exception as e:
        print("Error: ", e)

    # Show the frame
    cv2.imshow("Real-Time Engagement Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
