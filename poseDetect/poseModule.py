import cv2  #library
import mediapipe as mp #framework
import time


class poseDetector():
    def __init__(self,mode = False,complexity = 1,smooth_landmarks = True,enable_segmentations = False, smooth_segmentation = True, detectConf= 0.5,trackConf = 0.5):
        self.mode = mode    #mode of object of class is set to mode val
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentations = enable_segmentations
        self.smooth_segmentations =  smooth_segmentation
        self.detectConf = detectConf
        self.trackConf = trackConf                
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.complexity,self.smooth_landmarks,self.enable_segmentations,self.smooth_segmentations,self.detectConf,self.trackConf) #use default parameters

    def findPose(self,img,draw = True):
            #if draw == True

             #img is in BGR but mpPose works in RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)      
            
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img      
     
    #print(results.pose_landmarks)
    #Draw the points in realtime
    def markPosition(self,img,draw= True):
        self.landmarks=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx,cy= int(lm.x*w) , int (lm.y*h)  #to get pixel values
                self.landmarks.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)  
        return self.landmarks

def main():
    
    cap = cv2.VideoCapture('PoseVideo/1.mp4')  #Enter file path here of video to track movement in
    pTime = 0
    detector = poseDetector()
    while True:
        success,img = cap.read()
        img = detector.findPose(img)
        lmList = detector.markPosition(img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        #for placing the text 
        cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        cv2.imshow('Image',img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
if __name__=="__main__":
    main()