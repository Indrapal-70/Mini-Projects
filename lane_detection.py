import cv2 as cv
import numpy as np

cap = cv.VideoCapture('trafic.mp4')
while True:
    res, frame = cap.read()
    if not res:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0,height),
        (width, height),
        (width,height//2),
        (0, height//2)
    ]], np.int32)

    cv.fillPoly(mask, polygon,255)
    masked_edges = cv.bitwise_and(edges, mask)

    lines = cv.HoughLinesP(masked_edges, 1, np.pi/100, 50, maxLineGap=50)
    line_image = np.zeros_like(frame)
    
    if lines is not None :
        for line in lines :
            x1, y1, x2, y2 = line[0]
            cv.line(line_image, (x1,y1), (x2,y2), (0,0,255), thickness= 10)
    
    combo = cv.addWeighted(frame, 0.8, line_image, 1,1)
    cv.imshow('lane', combo)

    if cv.waitKey(30) & 0xFF == ord("f") : 
        break

cap.release()
cv.destroyAllWindows()