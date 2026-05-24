''' The application supports the following functionality:
* - Loading a video stream from disk or camera.
* - Initializing the YOLO detector with the desired model and labels.
* - Detecting objects within each frame of the video.
* - Drawing bounding boxes around detected objects and saving the result.

A few words about model formats:

* PyTorch .pt models: Full API support (.train, .predict, device selection when loading, etc.).

* ONNX models: Only .predict and .val are supported. Device is selected in the predict call, not in the constructor.
    You should use model.predict() for ONNX models.
    You cannot call .to(device) or use the model object as a callable (e.g., model(frame)) with ONNX models.
'''

import cv2
import threading
import queue
import time
import argparse

# Import the YOLO class from the ultralytics package
from ultralytics import YOLO

# Import module for drawing bounding boxes and annotations
from ultralytics.utils.plotting import Annotator

class SafeQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.finished = False
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def enqueue(self, item):
        with self.cond:
            self.queue.put(item)
            self.cond.notify()

    def dequeue(self):
        with self.cond:
            while self.queue.empty() and not self.finished:
                    self.cond.wait()
            if self.queue.empty() and self.finished:
                return None
            return self.queue.get()

    def set_finished(self):
        with self.cond:
            self.finished = True
            self.cond.notify_all()


# Let's define the drawing function for bounding boxes
def draw_bounding_boxes(frame, result, names):
    annotator = Annotator(frame)
    for box in result.boxes:
        b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        c = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{names[c]} {conf:.2f}"
        annotator.box_label(b, label)
    return annotator.result()


# Main function to handle video processing with YOLO
def main(video_path, model_path, use_gpu=True, output_path=None):
    device = 'cuda' if use_gpu else 'cpu'
    print(f"Using model: {model_path} on device: {device}")

    model = YOLO(model_path)
    model.to(device)
    names = model.names


    frame_queue = SafeQueue()
    processed_queue = SafeQueue()

    # Video capture thread
    def capture_thread():
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.enqueue(frame.copy())
        frame_queue.set_finished()
        cap.release()

    # Processing thread
    def process_thread():
        while True:
            frame = frame_queue.dequeue()
            if frame is None:
                break
            results = model(frame, verbose=False)[0]  # YOLO inference
            processed_queue.enqueue((frame, results))
        processed_queue.set_finished()

    # Display and write thread
    def display_thread():
        writer = None
        window_name = "YOLO Video Processing"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 320)
        while True:
            item = processed_queue.dequeue()
            if item is None:
                break
            frame, results = item
            frame = draw_bounding_boxes(frame, results, names)
            cv2.imshow(window_name, frame)
            if output_path:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
                writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                frame_queue.set_finished()
                processed_queue.set_finished()
                break
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Start threads
    threads = [
        threading.Thread(target=capture_thread),
        threading.Thread(target=process_thread),
        threading.Thread(target=display_thread)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="../data/video/pedtracking.mp4", help="Input video path")
    parser.add_argument("-m", "--model", default="../models/yolo11l.pt", help="YOLO model path")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-o", "--output", default="../data/experiment/yolo_processed.mp4", help="Output video path")
    args = parser.parse_args()

    main(args.video, args.model, not args.cpu, args.output)