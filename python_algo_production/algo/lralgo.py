#### Algotirthm Implementation for Largest Rectangle
import numpy as np

## Function of fiding the largest rectange under a histogram
def lr_hist_naive(hist):       # hist is an numpy array

    maxArea = 0;
    length = len(hist);

    # loop through the histogram
    for i, item in enumerate(hist):
        left = i
        right = i

        ## stretch towards left to seach
        while left > 0 and hist[left - 1] >= hist[i]:
            left -= 1
        ## stretch towards right to search
        while right < length-1 and hist[right + 1] >= hist[i]:
            right += 1

        ## update the area
        curArea = (right - left + 1) * hist[i]
        if curArea > maxArea:
            maxArea = curArea
            largest_left = left
            largest_right = right
            largest_bar = i
    # the following output will be captured by pytest to make the results clean
    # to enable the print: simply try pytest -s  (short for --capture=no)
    # can also use logger instead: import logging
    #  logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)
    # logger.info(f"Largest Rectangle: [{largest_left} <- {largest_bar} -> {largest_right}]")
    print(f"Largest Rectangle: [{largest_left} <- {largest_bar} -> {largest_right}]\n")
    return maxArea, largest_left, largest_bar, largest_right


## Optimal function of fiding the largest rectange under a histogram
# using a list to simulate the stack used for C++ implementation
def lr_hist(hist):       # hist is an numpy array

    maxArea = 0;
    sta = [];    # initialize a list to simulate a stack
    # loop through the histogram
    for i in range(len(hist) + 1):  # for i, item in enumerate(hist) wont reach the last element
        height = 0 if i == len(hist) else hist[i]  # C++: height = (i == hist.size())? 0:hist[i]
        ## make sure we have an extra iteration to cover the last bar

        ## step 1: check to remove the previous stalled taller (equivalent) bar
        while len(sta) != 0 and height < hist[sta[-1]]:
            topIndex = sta[-1]       # tallest bar prior to current bar
            sta.pop()                # remove last element

            ## deterine the width of the rectangle
            # If the stack is empty, then the rectangle spans from index 0 to i-1 because
            # the bar at that popped index is the smallest seen so far (from the beginning of the histogram)
            # Otherwise, it spans from the element after the new back to i - 1.
            width = i if len(sta) == 0 else i - 1 - sta[-1]
            priorArea = width * hist[topIndex]        # area until i-1

            # Use Ternary Operator When the condition is simple / no need for debug
            maxArea = priorArea if priorArea > maxArea else maxArea

        # step 2:push in current bar
        sta.append(i)

    return maxArea