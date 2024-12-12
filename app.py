from flask import Flask, request, jsonify
from cvzone.PoseModule import PoseDetector
import cv2
import numpy as np

app = Flask(__name__)

# Initialize the PoseDetector
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

@app.route('/')
def index():
    return app.send_static_file('index.html')  # Serve the static HTML file

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the image data from the request
    file = request.files['frame']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform pose detection
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=False)

    # Return keypoints or any pose detection results
    result = {
        "landmarks": lmList if lmList else [],
        "bounding_box": bboxInfo if bboxInfo else {}
    }
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
