import cv2#library
import mediapipe as mp #framework
import time
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose() #use default parameters
#alternates between tracking and detection
# self,
#                static_image_mode=False,  false for video and true for img
#                model_complexity=1,
#                smooth_landmarks=True,
#                enable_segmentation=False,
#                smooth_segmentation=True,
#                min_detection_confidence=0.5, #when tracking less than 0.5 it will detect
#                min_tracking_confidence=0.5 #when more than 0.5 it iwll track


cap = cv2.VideoCapture('PoseVideo/2.mp4')
pTime = 0
while True:
    success,img = cap.read()     #img is in BGR but mpPose works in RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)      
    print(results.pose_landmarks)
    #Draw the points in realtime 
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx,cy= int(lm.x*w) , (lm.y*h)  #to get pixel values
            cv2.circle(img,(cx,cy),4,(255,0,0),cv2.FILLED)
    #for fps calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #for placing the text 
    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
