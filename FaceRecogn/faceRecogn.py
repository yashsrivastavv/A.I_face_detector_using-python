import cv2
import os

alg ="F:\pythonPro\openCV\haarcascade_frontalface_default.xml" #Here to provide the location of haarcascade_frontalface (xml) file
haar = cv2.CascadeClassifier(alg)

cam= cv2.VideoCapture(0)

dataset="F:\pythonPro\openCV"  #locate the file in disk
name="captured_images"        #give the name of folder which will create a folder and store captured images
path = os.path.join(dataset,name)
if not os.path.isdir(path):
    os.mkdir(path)

(width,height)= (130,100)
count=1

while (count <30):
    print(count)
    _,img=cam.read()
    grayImg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        onlyFace = grayImg[y:y+h,x:x+w]
        resizeImg = cv2.resize(onlyFace,(width,height))
        cv2.imwrite("%s/%s.jpg" % (path,count),resizeImg)
        count+=1


    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
print("Face Captured Sucessfully")
cam.release()
cv2.destroyAllWindows()