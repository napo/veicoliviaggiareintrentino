import os
import cv2
import numpy as np
import areas

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

for img in os.listdir("frame_set"):
    next = False

    frame = cv2.imread(f"frame_set/{img}")

    area= areas.area[f"{img}"]

    while not next:
        cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
        cv2.putText(frame,str(img),(10,60),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        cv2.imshow("RGB", frame)

        if cv2.waitKey(0) & 0xFF == 27:
            break

frame.release()
cv2.destroyAllWindows()