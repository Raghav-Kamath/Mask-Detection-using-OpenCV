from cv2 import CascadeClassifier,VideoCapture
from cv2 import putText,rectangle,FONT_HERSHEY_SIMPLEX,LINE_AA
from cv2 import resize,imshow,destroyAllWindows,waitKey
from numpy import save
haar=CascadeClassifier('haarcascade_frontalface_default.xml')
cap=VideoCapture(0)
data=[]

while True:
    flag,img=cap.read()
    if flag:
        faces=haar.detectMultiScale(img,1.3,6)
        putText(img,"Make sure only your face is being detected",(20,30),FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),4,LINE_AA)
        putText(img,"Make sure only your face is being detected",(20,30),FONT_HERSHEY_SIMPLEX,0.75,(0,0,0),2,LINE_AA)
        putText(img,"Press Esc to force quit(use only in emergency!)",(20,60),FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),4,LINE_AA)
        putText(img,"Press Esc to force quit(use only in emergency!)",(20,60),FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2,LINE_AA)

        for x,y,w,h in faces:
            rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            putText(img,"Face Detected! "+ str(len(data))+"/200",(x,y),FONT_HERSHEY_SIMPLEX,0.75,(244,250,250),2)
            face=img[y:y+h,x:x+w,:]
            face=resize(face,(50,50))
            print(len(data))
            if len(data) < 200:
                data.append(face)
        imshow('Training without mask',img)
        if waitKey(2)==27 or len(data) >=200: # pressing esc will force quit so it is not recommended as we need 200 images
            print('Done Training without mask')
            break
cap.release()
destroyAllWindows()

save('without_mask.npy',data)
