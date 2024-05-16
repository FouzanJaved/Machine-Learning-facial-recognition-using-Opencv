import cv2
import numpy as np
#This is for client laptop to connect to robot bottle Server
import urllib2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner\\trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

ip="http://192.168.100.3:8080/"

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        print (Id)
        print conf
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)

        #the lower the value than 60 the higher the accuracy of recognition
        if(conf<70):
            if(Id<32):
                cv2.putText(im,"Farhan", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                contents = urllib2.urlopen(ip+"1").read()
                print contents
            elif(Id>31 & Id<62):
                cv2.putText(im,"Danish", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                contents = urllib2.urlopen(ip+"2").read()
                print contents
                
            elif(Id>61 & Id<92):
                cv2.putText(im,"Shahzad", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                contents = urllib2.urlopen(ip+"3").read()
                print contents

            else:
                cv2.putText(im,"Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                contents = urllib2.urlopen(ip+"5").read()
                print contents
                
            #else:
        else:
            cv2.putText(im,"Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        
       # if(conf<20):
       ##     if(Id==1):
        #        Id="fouzan"
                
       #         cv2.putText(im,"fouzan",(x,y-10),font,8,(0,255,0),1)
       #         print"fouzan"
                
      #      elif(Id==2):
       #         Id="Sam"
       #         cv2.putText(im,"Sam",(x,y-10),font,0.55,(255,0,0),1)
                
       # else:
      #      Id="Unknown"
       ##     cv2.putText(im,"unknown",(x,y-10),font,0.55,(0,0,255),1)
           
        
        
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
