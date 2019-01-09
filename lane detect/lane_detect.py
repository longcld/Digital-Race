import cv2
import numpy as np
import sys

#Convert image to Binary
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
    return img

#Transform image to bird view
def birth_view_transform(img):
    W_BirdView = rows
    H_BirdView = cols
    pts1 = np.float32([[0,skyline],[cols,skyline],[cols, rows],[0,rows]])
    pts2 = np.float32([[0, 0], [W_BirdView - 100, 0], [W_BirdView - 100, H_BirdView], [100, H_BirdView]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = img[skyline:H_BirdView]
    dst = cv2.warpPerspective(img, M, dst.shape[0:2], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    return dst

def fill_lane(img):
    copy = img.copy()
    if (len(copy.shape) == 3):
        img_processed = preProcess(copy)
    else:
        img_processed = copy
    lines = cv2.HoughLinesP(img_processed, rho=1, theta=np.pi/180, threshold=10, minLineLength = 1, maxLineGap = 20)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(copy, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return copy

def getIntersection(line1, line2):
    s1 = np.array(line1[0])
    e1 = np.array(line1[1])

    s2 = np.array(line2[0])
    e2 = np.array(line2[1])

    a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
    b1 = s1[1] - (a1 * s1[0])

    a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
    b2 = s2[1] - (a2 * s2[0])

    if abs(a1 - a2) < sys.float_info.epsilon:
        return False

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return (int(x), int(y))

def update(img):
    #Read Image
    #img = cv2.imread("test2.png")
    #img = cv2.resize(img, (320, 240))
    
    rows, cols, ch = img.shape
    skyline = int(rows*1.2/3)
    
    #Create Windows
    cv2.namedWindow("Before", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Before", 320*2, 240*2)
    cv2.namedWindow("After", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("After", 320*2, 240*2)
    
    #Processing
    binary_img = preProcess(fill_lane(img))
    bird_view = fill_lane(birth_view_transform(binary_img))
    
    #Find the line of left and right lane
    xleft_1, xright_1, yleft_1, yright_1 = 2, cols-1, 0, 0
    xleft_2, xright_2, yleft_2, yright_2 = 3, cols - 11, 0, 0
    
    for i in range (210, 140, -1):
        if (binary_img[i][xleft_1] == 255) and (yleft_1 == 0):
            yleft_1 = i
        if (binary_img[i][xright_1] == 255) and (yright_1 == 0):
            yright_1 = i
        if (binary_img[i][xleft_2] == 255) and (yleft_2 == 0):
            yleft_2 = i
        if (binary_img[i][xright_2] == 255) and (yright_2 == 0):
            yright_2 = i
        if (yleft_1 != 0) and (yright_1 != 0) and (yleft_2 != 0) and (yright_2 != 0):
            break
            
    #Two lines
    left_line = [(xleft_1,yleft_1), (xleft_2,yleft_2)]
    right_line = [(xright_1, yright_1), (xright_2,yright_2)]

    #Find intersection of two lines
    intersection = getIntersection(list(left_line), list(right_line))
        
    #Draw two lines
    cv2.line(img, (xleft_1, yleft_1), intersection, (0, 255, 0), 2)
    cv2.line(img, (xright_1, yright_1), intersection, (0, 255, 0), 2)
    
    cv2.line(img, (xleft_1, yleft_1), (xleft_2, yleft_2), (0, 255, 0), 2)
    cv2.line(img, (xright_1, yright_1), (xright_2, yright_2), (0, 255, 0), 2)

    #Get the angle
    check = intersection[0] - cols/2
    x = abs(intersection[0] - cols/2)
    y = abs(intersection[1] - rows)
    angle = np.degrees(np.arctan(x/y))
    
    #Show results
    #cv2.imshow("Before", binary_img)
    #cv2.imshow("After", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    if check < 0:
        return -angle
    else:
        return angle
