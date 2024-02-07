from cv2 import CascadeClassifier,VideoCapture,FONT_HERSHEY_COMPLEX,putText,rectangle,resize,imshow,waitKey,destroyAllWindows,LINE_AA
from numpy import load,r_,zeros
from sklearn.metrics._pairwise_distances_reduction import _datasets_pair,_middle_term_computer

haar=CascadeClassifier('haarcascade_frontalface_default.xml')
with_mask=load('with_mask.npy')
without_mask=load('without_mask.npy')

with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)

X=r_[with_mask,without_mask]

labels =zeros(X.shape[0])
labels[200:]=1.0

names={0:'Mask',1:'No Mask'}

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.25)
svm=SVC(kernel='linear')
svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)
# print(accuracy_score(y_test,y_pred))

cap=VideoCapture(0)
data=[]
font=FONT_HERSHEY_COMPLEX
while True:
    flag,img=cap.read()
    if flag:
        faces=haar.detectMultiScale(img,1.3,6)
        putText(img,"Press Esc to Close",(10,30),FONT_HERSHEY_COMPLEX,0.75,(255,255,255),4,lineType=LINE_AA)
        putText(img,"Press Esc to Close",(10,30),FONT_HERSHEY_COMPLEX,0.75,(0,0,0),2,lineType=LINE_AA)
        for x,y,w,h in faces:
            rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=resize(face,(50,50))
            face=face.reshape(1,-1)
            pred=svm.predict(face)[0]
            n=names[int(pred)]
            putText(img,n,(x,y),font,1,(244,250,250),2)
            # print(n)
        imshow('Mask Detection',img)
        if waitKey(2)==27:
            break
cap.release()
destroyAllWindows()
