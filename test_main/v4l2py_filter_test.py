import cv2
import numpy as np
from python_algo_production.lib import v4l2_py  ## v4l2_py.so with module name v4l2_py: built by pybind11 and C++ (including cudart)
from python_algo_production.lib.image_filter_py import ImageFilter, KERNEL_15x15, KERNEL_3x3, KERNEL_7x7

# Constants
WIDTH = 1920
HEIGHT = 1080
DEVICE = "/dev/video0"
USE_PITCHED_MEM = True
AVG_FILTER = False
KERNEL_WIDTH = 15

def main():
    # Initialize V4L2 camera
    camera = v4l2_py.V4l2Capture(DEVICE, WIDTH, HEIGHT, v4l2_py.IO_METHOD.USERPTR, True)
    num_channels = camera.get_num_channels()
    # display camera status
    camera.print_v4l2_status()

    # Dummy CUDA filter class — you would wrap your C++ ImageFilter class instead
    #  ImageFilter filter(WIDTH, HEIGHT, numChannels, pitchedMem, resultImg, sharedMem);
    filter = ImageFilter(WIDTH, HEIGHT, num_channels, USE_PITCHED_MEM, True, True)
    filter.print_imagefilter_status()

    # Allocate memory for processed frame if using pitched
    processed_frame = np.zeros((HEIGHT, WIDTH, 2), dtype=np.uint8) if USE_PITCHED_MEM else None

    print("Starting capture... Press ESC to exit.")
    while True:
        frame = camera.get_frame()

        # Apply filter
        if KERNEL_WIDTH == 15:
            if AVG_FILTER:
                if USE_PITCHED_MEM:
                    filter.image_average15x15_pitch(processed_frame, frame)
                else:
                    filter.image_average15x15(frame)
            else:
                # kernel is a global define not tied to the filter object
                filter.load_filter_kernel(KERNEL_15x15, 15)
                if USE_PITCHED_MEM:
                    filter.image_filter15x15_pitch(processed_frame, frame)
                else:
                    filter.image_filter15x15(frame)

        # Convert YUYV to BGR for display
        yuyv = processed_frame if USE_PITCHED_MEM else frame
        yuyv_mat = np.frombuffer(yuyv, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
        bgr = cv2.cvtColor(yuyv_mat, cv2.COLOR_YUV2BGR_YUYV)

        cv2.imshow("V4L2 Capture + CUDA Filter", bgr)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Exiting.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
