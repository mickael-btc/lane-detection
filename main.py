from mss import mss
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

class DevNull:
    def write(self, msg):
        pass

def average_slope_intercept(image, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    try:
        # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
        left_line = make_coordinate(image, left_avg)
        right_line = make_coordinate(image, right_avg)
        return np.array([left_line, right_line])
    except Exception as e:
        return None

def make_coordinate(image, line_param):
    slope, intercept = line_param
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = image.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 300)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def draw_lines(image,lines):
    try:
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
        return image
    except:
        return image

def lane_zone(image):
    height = image.shape[0]
    triangle = np.array([[(0, height), (1600, height), (800, 800)]])
    # triangle = np.array([[(0, height), (0, 800), (1600, 800), (1600, height)]])
    # triangle = np.array([[
    #     (0, height),
    #     (0, 1000),
    #     (800, 700),
    #     (1600, 1000),
    #     (1600, height)
    # ]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def process_gray_blur_canny_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([23, 42, 133], dtype = "uint8")
    upper_yellow = np.array([40, 150, 255], dtype="uint8")

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 183, 255)
    
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)

    blur = cv2.GaussianBlur(mask_yw_image, (5,5), 0)
    # canny =  cv2.Canny(blur, 50, 150)

    return blur

def image_task(image):
    gray_image = process_gray_blur_canny_image(image)
    cropped_image = lane_zone(gray_image)
    lines = cv2.HoughLinesP(cropped_image, rho=2.5, theta=np.pi / 300, threshold=100, lines=np.array([]), minLineLength=100, maxLineGap=10)
    if lines is not None:
        new_lines = average_slope_intercept(image, lines)
        draw_lines(image, new_lines)
    return image
    # return gray_image
    # return cropped_image

def main():
    sys.stderr = DevNull()

    first_window = True

    with mss() as sct:
        monitor = {
            'top': 0,
            'left': 0,
            'width': 800,
            'height': 640
        }
        screen = np.array(sct.grab(monitor))

    while(True):
        if first_window:
            cv2.namedWindow("output", cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow("output", 800, 640)
            cv2.imshow("output", screen)
            first_window = False

        screen = np.array(sct.grab(monitor))
        cv2.imshow("output", image_task(screen))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()