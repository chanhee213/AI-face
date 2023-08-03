import cv2
import threading
import numpy as np
from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
import cv2, glob, dlib
import time

overlay_text = ''
x1 = 0
y1 = 0
isFound = False

def run():
    global frame
    global webcam
    global overlay_text
    global x1
    global y1
    global isFound
    global img2
    global img3
    global analyzed

    overlay_text_prev = ''

    
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    while webcam.isOpened():
        status, frame = webcam.read()
        img2 = frame 
        if status:
            if isFound:
                print('check in the thread..'+overlay_text)

                crop = frame[x1,y1]


                img2 = frame + img2                  
            cv2.imshow("preview", img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()

def main_func():
    global frame
    global webcam
    global overlay_text
    global x1
    global y1
    global isFound
    global model
    global img1 
    global img2
    model = VGGFace.loadModel()

  
    img1 = cv2.imread('./face/cloud.png')
    img1 = cv2.resize(img1,(640,480))

    webcam = cv2.VideoCapture(0)
    
    thread = threading.Thread(target=run)

    #if webcam.isOpened():
    #    print('웹캠 캡쳐중!')

    thread.start()

    begin = time.time()

    while webcam.isOpened():
        end = time.time()
        #시간차
        result = end - begin
        #여기서 round는 파이썬에서 소수점 자리수 조절에 활용됩니다.
        if result > 5:
            begin = time.time()
            result = 0
            try :
                result = DeepFace.analyze(frame, actions = ['age', 'gender', 'race', 'emotion'])
                emotion = result[0]['dominant_emotion']
                x1 = result[0]['region']['x']
                y1 = result[0]['region']['y']
                
                isFound = True
            except:
                pass
        elif result > 4:
            isFound = False
            

    thread.join()
    print("Bye :)")

if __name__ == "__main__":
    main_func()
    