import os
import cv2

# run the program
# click on the image to place the points of the area
# press ESC to skip to next frame
# if you don't place any point the area will be the entire frame
# the areas will be saved in areas.py

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = (x, y)
        param.append(refPt)

cv2.namedWindow('RGB')

instr1 = "Click with the cursor to place the points of the area."
instr2 = "Press ESC to skip to next the frame."

for img in os.listdir("frame_set"):
    next = False

    frame = cv2.imread(f"frame_set/{img}")

    area= []

    cv2.setMouseCallback('RGB', RGB, area)

    while not next:
        cv2.rectangle(frame, (5, 10), (570, 55), (255, 255, 255), -1)
        cv2.putText(frame,str(instr1),(10,30),cv2.FONT_HERSHEY_PLAIN,1.2,(0,0,0),2)
        cv2.putText(frame,str(instr2),(10,50),cv2.FONT_HERSHEY_PLAIN,1.2,(0,0,0),2)
        cv2.imshow("RGB", frame)

        if cv2.waitKey(0) & 0xFF == 27:
            break
    
    if len(area) == 0:
        area = [(0,0),(frame.shape[0],0),(frame.shape[0],frame.shape[1]),(0,frame.shape[1])]

    print(area)
    
    with open('areas.py', 'a') as file:
        file.write(f'area["{img}"] = {str(area)}\n')

frame.release()
cv2.destroyAllWindows()