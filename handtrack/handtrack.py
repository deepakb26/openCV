import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
#
mpHands = mp.solutions.hands
#
hands = mpHands.Hands()
#
mpDraw = mp.solutions.drawing_utils
# TO DISPLAY FPS
pTime = 0
cTime = 0
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x * w), int(lm.y*h)
                
                cv2.circle(img,(cx,cy),20,(255,0,255),cv2.FILLED) 


            #Draw landmark points on hands for real time image.
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) 
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #Displays FPS as an overlay over the image
    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),3)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
  
