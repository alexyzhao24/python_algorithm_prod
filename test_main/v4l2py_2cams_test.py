import threading
import multiprocessing
import numpy as np
import traceback
import cv2
from python_algo_production.lib import v4l2_py
import time

'''
Mult-processing is truly in parallel: demonstrated when the two camera is HD resolution while
   * multithreading display is lagging and jittering
   * multiprocessing display is smooth and fast

Using multiprocessing to replace multi-threading to break the GIL issue in Python
   * Use multiprocessing.Process instead of threading.Thread.
   * Use multiprocessing.Manager() to create shared memory objects like your frames list and locks.
   * Replace threading.Lock with multiprocessing.Lock.
   * Use a multiprocessing.Event for signaling stop.

* Each camera capture runs in its own thread/process
* Shared frames list stores the latest BGR frame from each camera.
* The main thread reads frames and shows combined view.
* Press ESC to quit cleanly.

About parallelism in multithreading:
* Yes, your threads are concurrent and effectively "parallel enough" for video capture,
  because the capture calls are done in native code that releases the GIL (Global Interpreter Lock)
* No, they’re not running in parallel Python bytecode on multiple CPU cores at the same time, so
  rendering and frame combining happen sequentially
'''

WIDTH, HEIGHT = 1920, 1080
IO_METHOD = v4l2_py.IO_METHOD.MMAP
CAMERA1 = "/dev/video4"
CAMERA2 = "/dev/video0"  # change this to your second camera device

MULTI_THREAD = False

### thread per camera for capturing frame runs in parallel
def capture_loop(device, frames, index, frame_lock, stop_event):

    # the following camera initialization is outside the while loop, so only runs once
    try:
        cap = v4l2_py.V4l2Capture(device, WIDTH, HEIGHT, IO_METHOD)
        cap.print_v4l2_status()
    except Exception as e:
        print(f"[ERROR] Failed to open {device}: {e}")
        traceback.print_exc()
        stop_event.set()  # # Turns the flag to True — tells all threads to stop
        return

    # the following capture of frame is repeated!
    while not stop_event.is_set(): # stop_event.is_set() returns False (unless set)
        try:
            frame_buf = cap.get_frame()
            yuyv = np.frombuffer(frame_buf, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)

            # Store the captured frame thread-safely
            with frame_lock:
                frames[index] = bgr.copy()  # copy to prevent concurrent modification
        except Exception as e:
            print(f"[ERROR] Capture failed on {device}: {e}")
            traceback.print_exc()
            stop_event.set()  # Turns the flag to True — tells all threads to stop
            break


###  Display thread pulls latest frames with mutex protection
def display_loop(frames, frame_lock, stop_event):

    # the following capture of frame is repeated!
    while not stop_event.is_set():
        start = time.time()
        with frame_lock:
            # 'else np.zeros()' is critical for valid later stament 'combined = np.hstack((f0, f1))!
            f0 = frames[0].copy() if frames[0] is not None else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            f1 = frames[1].copy() if frames[1] is not None else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            combined = np.hstack((f0, f1))
            cv2.imshow("Two Cameras Side by Side", combined)

        if cv2.waitKey(1) == 27:  # ESC to quit
            stop_event.set()
            break

    cv2.destroyAllWindows()


def main():
    if MULTI_THREAD:
        frames = [None, None]
        frame_lock = threading.Lock()
        # creates an Event object with an internal boolean flag that is initially False.
        stop_event = threading.Event()

        # Create capture threads
        cap1 = threading.Thread(target=capture_loop, args=(CAMERA1, frames, 0, frame_lock, stop_event), daemon=True)
        cap2 = threading.Thread(target=capture_loop, args=(CAMERA2, frames, 1, frame_lock, stop_event), daemon=True)

        # Create display thread
        # Notice that we need pass in a tuple args=(stop_event, ) with an extra ","
        # args=(stop_event) just a variable, like, args=stop_event, not a tuple!
        display = threading.Thread(target=display_loop, args=(frames, frame_lock, stop_event))
    else:
        manager = multiprocessing.Manager()
        frames = manager.list([None, None])  # shared list
        frame_lock = manager.Lock()
        stop_event = multiprocessing.Event()

        cap1 = multiprocessing.Process(target=capture_loop, args=(CAMERA1, frames, 0, frame_lock, stop_event))
        cap2 = multiprocessing.Process(target=capture_loop, args=(CAMERA2, frames, 1, frame_lock, stop_event))
        display = multiprocessing.Process(target=display_loop, args=(frames, frame_lock, stop_event))


    ## syntax for multi-threading and mult-processing are the same here
    cap1.start()
    cap2.start()
    display.start()

    # Wait for clean exit
    display.join()
    cap1.join()
    cap2.join()


if __name__ == "__main__":
    if MULTI_THREAD:
        print("Using Mutithreading")
    else:
       print("Using Mutiprocessing")

    main()
