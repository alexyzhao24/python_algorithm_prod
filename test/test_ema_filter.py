import test
import random
import sys
import matplotlib.pyplot as plt
from python_algo_production.signal.ema_filter import *

show_animation = True
pause_time = 0.001
duration = 10.0  # [s] duration of the signal
rate = 10.0  # [Hz] sampling rate, default 5 Hz, should be at least twice the cutoff frequency per Nyquist theorem
f_c = 1.0  # [Hz] cutoff frequency for the filter
amplitude = 5.0  # [m] amplitude of the cosine wave

# start and goal position
t0 = 0.0  # [s]
x0 = 10.0  # [m]
y0 = 10.0  # [m]

# function to generate a noise cosine wave: x(t) = x0 + A⋅cos(2π *fc*t)
def genereate_cosine_wave(x0, t0, rate, duration, amplitude=5.0, f_c=1.0):
    n_samples = int(rate * duration)
    t_values = [t0 + i / rate for i in range(n_samples)]
    clean_signal = [x0 + amplitude * np.cos(2 * np.pi * f_c * t) for t in t_values]
    noisy_signal = [x + random.uniform(-amplitude/2, amplitude/2) for x in clean_signal]
    return t_values, clean_signal, noisy_signal

# function to generate meaningful 2D noise points on a plane with time stamps
# when plotted with timestamp, it will look like a circular motion with some noise
def genereate_2d_noise_points(x0, y0, t0, rate, duration, amplitude=5.0, f_c=1.0):
    n_samples = int(rate * duration)
    t_values = [t0 + i / rate for i in range(n_samples)]
    clean_signal_x = [x0 + amplitude * np.cos(2 * np.pi * f_c * t) for t in t_values]
    clean_signal_y = [y0 + amplitude * np.sin(2 * np.pi * f_c * t) for t in t_values]
    noisy_signal_x = [x + random.uniform(-amplitude/2, amplitude/2) for x in clean_signal_x]
    noisy_signal_y = [y + random.uniform(-amplitude/2, amplitude/2) for y in clean_signal_y]
    return t_values, clean_signal_x, noisy_signal_x, clean_signal_y, noisy_signal_y

# simulate cosine wave with some noise
t_list, x_true_list, x_list = genereate_cosine_wave(x0, t0, rate, duration)

# simulate circular motion with some noise
t_list_xy, true_list_x, list_x, true_list_y, list_y = genereate_2d_noise_points(x0, y0, t0, rate, duration)

# display 1D signal for comparison
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# display 2D signal for comparison: Create a figure and two 3D subplots side-by-side
fig3d, axes3d = plt.subplots(2, 1, subplot_kw={'projection':'3d'}, figsize=(8, 12))

# non-blocking show to open window
plt.show(block=False)


def test_one_ema_filter_step():
    print(__file__ + " start!!")

    # create the filter
    alpha = 0.125
    one_ema_filter = OneEmaFilter(x0, alpha)

    if show_animation:
        axes[0].cla()
        axes[0].set_title("One EMA Filter Step Test")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Value")
        # let's fix the axis for better visualization
        axes[0].set_xlim(t0, t0 + duration)
        axes[0].set_ylim(x0 - 1.5*amplitude, x0 + 1.5*amplitude)
        # list for display purpose
        t_disp = []
        x_disp = []
        x_hat_disp =[]
        x_true_disp = []

    ## step b step filtering
    for i in range(len(t_list)):
        t = t_list[i]
        x = x_list[i]
        x_true = x_true_list[i]
        x_hat = one_ema_filter.filter_step(x)

        if show_animation:
            t_disp.append(t)
            x_disp.append(x)
            x_hat_disp.append(x_hat)
            x_true_disp.append(x_true)
            # let's plot with line connecting the dots
            axes[0].plot(t_disp, x_disp, marker='x', color='red', linestyle='-')
            axes[0].plot(t_disp, x_hat_disp, marker='o', color='blue', linestyle='-')
            axes[0].plot(t_disp, x_true_disp, marker='^', color='green', linestyle='-')

            plt.draw()
            plt.pause(pause_time)
        else:
            print(f"Time:{t:.2f}, Value:{x:.2f} -> Filtered Value:{one_ema_filter.x_hat:.2f}")

    print("Test completed successfully.")


def test_one_ema_filter():
    print(__file__ + " start!!")

    # create the filter
    alpha = 0.125
    one_ema_filter = OneEmaFilter(x0, alpha)

    if show_animation:
        axes[1].cla()
        axes[1].set_title("One EMA Filter Test")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Value")
        # let's fix the axis for better visualization
        axes[1].set_xlim(t0, t0 + duration)
        axes[1].set_ylim(x0 - 1.5*amplitude, x0 + 1.5*amplitude)
        # list for display purpose
        t_disp = []
        x_disp = []
        x_hat_disp =[]
        x_true_disp = []


    # convert lists to numpy for filter processing
    t_array = np.array([])
    x_array = np.array([])
    for i in range(len(t_list)):
        t_array = np.append(t_array, t_list[i])
        x_array = np.append(x_array, x_list[i])

    # apply the filter
    x_hat_array = one_ema_filter.filter(x_array)

    if show_animation:
        for i in range(len(t_array)):
            t = t_array[i]
            x = x_array[i]
            x_true = x_true_list[i]
            x_hat = x_hat_array[i]

            t_disp.append(t)
            x_disp.append(x)
            x_true_disp.append(x_true)
            x_hat_disp.append(x_hat)
            # let's plot with line connecting the dots
            axes[1].plot(t_disp, x_disp, marker='x', color='red', linestyle='-')
            axes[1].plot(t_disp, x_hat_disp, marker='o', color='blue', linestyle='-')
            axes[1].plot(t_disp, x_true_disp, marker='^', color='green', linestyle='-')

            plt.draw()
            plt.pause(pause_time)
    else:
        ## not a lambd function, rather a combination of String's join() method, list comprehension and zip()
        print("\n".join([f"Time:{t:.2f}, Value:{x:.2f} -> Filtered Value:{x_hat:.2f}"
                         for t, x, x_hat in zip(t_array, x_array, x_hat_array)]))

    print("Test completed successfully.")


def test_one_ema_filter_params():
    print(__file__ + " start!!")

     # different alpha values for testing: large alpha values will have smaller lag, less smoothing
    alpha = [0.05, 0.125, 0.25, 0.5, 0.75, 0.95]
    # create the filters with different alpha values
    one_ema_filter1 = OneEmaFilter(x0, alpha[0])
    one_ema_filter2 = OneEmaFilter(x0, alpha[1])
    one_ema_filter3 = OneEmaFilter(x0, alpha[2])
    one_ema_filter4 = OneEmaFilter(x0, alpha[3])
    one_ema_filter5 = OneEmaFilter(x0, alpha[4])
    one_ema_filter6 = OneEmaFilter(x0, alpha[5])

    # Put them in a list for easier indexing:
    one_ema_filters = [
            one_ema_filter1,
            one_ema_filter2,
            one_ema_filter3,
            one_ema_filter4,
            one_ema_filter5,
            one_ema_filter6
        ]

    # display tests in comparison with different parameters
    param_fig, param_axes = plt.subplots(len(alpha), figsize=(10, 24))

    if show_animation:
        t_disp = []
        x_disp = []
        x_hats_disp =[[] for _ in range(len(alpha))]
        x_true_disp = []
        for i in range(len(alpha)):
            param_axes[i].cla()
            param_axes[i].set_title(f"One EMA Filter with alpha={alpha[i]}")
            param_axes[i].set_xlabel("Time [s]")
            param_axes[i].set_ylabel("Value")
            # let's fix the axis for better visualization
            param_axes[i].set_xlim(t0, t0 + duration)
            param_axes[i].set_ylim(x0 - 1.5*amplitude, x0 + 1.5*amplitude)

    # convert lists to numpy for filter processing
    t_array = np.array([])
    x_array = np.array([])
    for i in range(len(t_list)):
        t_array = np.append(t_array, t_list[i])
        x_array = np.append(x_array, x_list[i])

    # apply the filters
    x_hats_array = [np.array([]) for _ in range(len(alpha))]
    for i in range(len(alpha)):
        x_hats_array[i] = one_ema_filters[i].filter(x_array)

    if show_animation:
        for i in range(len(t_array)):
            t = t_array[i]
            x = x_array[i]
            x_true = x_true_list[i]
            for j in range(len(alpha)):
                x_hats_disp[j].append(x_hats_array[j][i])
            t_disp.append(t)
            x_disp.append(x)
            x_true_disp.append(x_true)

            # let's plot with line connecting the dots
            for j in range(len(alpha)):
                param_axes[j].plot(t_disp, x_disp, marker='x', color='red', linestyle='-')
                param_axes[j].plot(t_disp, x_hats_disp[j], marker='o', color='blue', linestyle='-')
                param_axes[j].plot(t_disp, x_true_disp, marker='^', color='green', linestyle='-')

            plt.draw()
            plt.pause(pause_time)

        # blocking show to open window
        plt.show(block=True)
    else:
        ## not a lambd function, rather a combination of String's join() method, list comprehension and zip()
        print("\n".join([f"Time:{t:.2f}, Value:{x:.2f} -> Filtered Values:{', '.join([f'{x_hat:.2f}' for x_hat in x_hats])}"
                         for t, x, *x_hats in zip(t_array, x_array, *x_hats_array)]))

    print("Test completed successfully.")


def test_two_ema_filter_step():
    print(__file__ + " start!!")

    # create the filter
    alpha = 0.125
    two_ema_filter = TwoEmaFilter(x0, y0, alpha)

    axes3d[0].cla()
    axes3d[0].set_title("Two EMA Filter Step Test")
    axes3d[0].set_xlabel("Time [s]")
    axes3d[0].set_ylabel("X")
    axes3d[0].set_zlabel("Y")

    # let's fix the axis for better visualization
    axes3d[0].set_xlim(t0, t0 + duration)
    axes3d[0].set_ylim(x0 - 1.5*amplitude, x0 + 1.5*amplitude)
    axes3d[0].set_zlim(y0 - 1.5*amplitude, y0 + 1.5*amplitude)

    # list for display purpose
    t_disp = []
    x_disp = []
    x_hat_disp =[]
    x_true_disp = []
    y_disp = []
    y_hat_disp =[]
    y_true_disp = []

    ## step b step filtering
    for i in range(len(t_list_xy)):
        t = t_list_xy[i]
        xy = [list_x[i], list_y[i]]
        x_true = true_list_x[i]
        y_true = true_list_y[i]

        xy_hat = two_ema_filter.filter_step(xy)

        t_disp.append(t)
        x_disp.append(xy[0])
        x_hat_disp.append(xy_hat[0])
        x_true_disp.append(x_true)
        y_disp.append(xy[1])
        y_hat_disp.append(xy_hat[1])
        y_true_disp.append(y_true)

        # let's plot with line connecting the dots
        axes3d[0].plot(t_disp, x_disp, y_disp, color='red', linestyle='-')
        axes3d[0].plot(t_disp, x_hat_disp, y_hat_disp,  color='blue', linestyle='-')
        axes3d[0].plot(t_disp, x_true_disp, y_hat_disp,  color='green', linestyle='-')

        plt.draw()
        plt.pause(pause_time)
    print("Test completed successfully.")


def test_two_ema_filter():
    print(__file__ + " start!!")

    # create the filter
    alpha = 0.125
    two_ema_filter = TwoEmaFilter(x0, y0, alpha)

    axes3d[1].cla()
    axes3d[1].set_title("Two EMA Filter Test")
    axes3d[1].set_xlabel("Time [s]")
    axes3d[1].set_ylabel("X")
    axes3d[1].set_zlabel("Y")

    # let's fix the axis for better visualization
    axes3d[1].set_xlim(t0, t0 + duration)
    axes3d[1].set_ylim(x0 - 1.5*amplitude, x0 + 1.5*amplitude)
    axes3d[1].set_zlim(y0 - 1.5*amplitude, y0 + 1.5*amplitude)

    # list for display purpose
    t_disp = []
    x_disp = []
    x_hat_disp =[]
    x_true_disp = []
    y_disp = []
    y_hat_disp =[]
    y_true_disp = []


    # convert lists to numpy for filter processing
    t_array = np.array([])
    # Start with empty array with shape (0, 2): expand to shape (n, 2)
    xy_array = np.empty((0, 2))
    for i in range(len(t_list_xy)):
        t_array = np.append(t_array, t_list[i])
        xy_array = np.append(xy_array, np.array([[list_x[i], list_y[i]]]), axis=0) # shape (n, 2)

    # apply the filter
    xy_hat_array = two_ema_filter.filter(xy_array)

    for i in range(len(t_array)):
        t = t_array[i]
        x = xy_array[i][0]
        x_true = true_list_x[i]
        x_hat = xy_hat_array[i][0]
        y = xy_array[i][1]
        y_true = true_list_y[i]
        y_hat = xy_hat_array[i][1]

        t_disp.append(t)
        x_disp.append(x)
        x_true_disp.append(x_true)
        x_hat_disp.append(x_hat)
        y_disp.append(y)
        y_true_disp.append(y_true)
        y_hat_disp.append(y_hat)

        # let's plot with line connecting the dots
        axes3d[1].plot(t_disp, x_disp, y_disp, color='red', linestyle='-')
        axes3d[1].plot(t_disp, x_hat_disp, y_hat_disp, color='blue', linestyle='-')
        axes3d[1].plot(t_disp, x_true_disp, y_true_disp, color='green', linestyle='-')

        plt.draw()
        plt.pause(pause_time)


    # blocking show to open window
    plt.show(block=True)
    print("Test completed successfully.")
