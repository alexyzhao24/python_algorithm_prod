import cv2
import numpy as np
from python_algo_production.lib import v4l2_py  ## v4l2_py.so with module name v4l2_py: built by pybind11 and C++ (including cudart)

'''
To run this script from terminal with pyest from project root:
> PYTHONPATH=test_main pytest test_main/v4l2py_test.py
'''

# Initialize v4l2 capture
io_method = v4l2_py.IO_METHOD.MMAP
cap = v4l2_py.V4l2Capture("/dev/video0", 640, 480, io_method)

# display camera status
cap.print_v4l2_status()

while True:
    frame_buf = cap.get_frame()  # NumPy array (YUYV)

    # Convert from YUYV to BGR
    yuyv = np.frombuffer(frame_buf, dtype=np.uint8).reshape((480, 640, 2))
    bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)

    cv2.imshow("Live", bgr)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cv2.destroyAllWindows()