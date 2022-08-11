import cv2
import torch
import numpy as np
import io
import os
import json
from PIL import Image

from flask import Flask, render_template, Response

app = Flask(__name__)

cap = cv2.VideoCapture(0)

# Load model
model = torch.hub.load("ultralytics/yolov5", 'custom', path='yolov5s.pt', force_reload=True, autoshape=True)  # force_reload = recache latest code
model.eval()

def predict():
    while cap.isOpened():
        
        sucess, frame = cap.read()

        if sucess == True:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, 640)
            results.render()
            results.print()
            img = np.squeeze(results.render())
            img_BGR = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            # Make detections
            # results = model(frame)

            # Display
            cv2.imshow('YOLO', np.squeeze(results.render()))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            break

        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\r')

    # cap.release()
    # cv2.destroyAllWindows()
    return

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(predict(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)