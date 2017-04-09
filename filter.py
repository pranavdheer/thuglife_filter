import dlib 
import cv2
import numpy as np

def Filter():
 image=cv2.imread('image.jpeg')
 a=image.copy()
 kernel = np.ones((5,5),np.uint8)
 a = cv2.erode(image,kernel,iterations = 1)
 a=a[:700,:,:]
 glasses=a[180:390,140:660]
 smoke=a[450:560,400:680]
 life=a[610:670,200:600]
 return glasses,smoke,life

detector  =  dlib.get_frontal_face_detector() 
predictor =  dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")




test_thug,smoke,life=Filter()
camera=cv2.VideoCapture(0)



while(1):
 ret,image=camera.read()
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
 dets = detector(gray, 1) 
 for k, d in enumerate(dets):
  face=image[d.top():d.bottom(),d.left():d.right(),:]


  shape = predictor(gray, d) 
  
  eyes = [shape.part(0).x,shape.part(16).x,shape.part(19).y,shape.part(30).y]
  lips = [shape.part(66).x,shape.part(54).x,shape.part(14).y,shape.part(57).y]
  text=  [shape.part(1).x,shape.part(15).x,shape.part(5).y,shape.part(8).y]
  
  crop_eyes=image[eyes[2]:eyes[3],eyes[0]:eyes[1]] 
  crop_lips=image[lips[2]:lips[3],lips[0]:lips[1]]
  crop_text=image[text[2]:text[3],text[0]:text[1]]
  
  test_thug_dis=cv2.resize(test_thug,(crop_eyes.shape[1],crop_eyes.shape[0]))
  smoke_dis=cv2.resize(smoke,(crop_lips.shape[1],crop_lips.shape[0]))
  life_dis=cv2.resize(life,(crop_text.shape[1],crop_text.shape[0]))

  d_eye=np.where(test_thug_dis>=200,crop_eyes,test_thug_dis) 
  d_lip=np.where(smoke_dis>=200,crop_lips,smoke_dis) 
  d_text=np.where(life_dis>=200,crop_text,life_dis) 
  
  image[eyes[2]:eyes[3],eyes[0]:eyes[1]]=d_eye 
  image[lips[2]:lips[3],lips[0]:lips[1]]=d_lip
  image[text[2]:text[3],text[0]:text[1]]=d_text 
 cv2.imshow("preview", image)
 
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
camera.release()
cv2.destroyAllWindows()
