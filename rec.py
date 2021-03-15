# -*- coding: UTF-8 -*-
import sys,os,dlib,glob,numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from skimage import io
import cv2
import imutils
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
REAL_THRESHOLD = 0.8 #will return fake if pred of real doesnt exceed threshold
std_correct_time=0

# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"

# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 比對人臉圖片資料夾名稱
faces_folder_path = "./rec"

# 需要辨識的人臉圖片名稱
#img_path = sys.argv[ 1]

# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()

# 載入人臉特徵點檢測器
sp = dlib.shape_predictor(predictor_path)

# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表
candidate = []

#選擇第一隻攝影機
cap = cv2.VideoCapture( 0)
#調整預設影像大小，預設值很大，很吃效能
cap.set(cv2. CAP_PROP_FRAME_WIDTH, 650)
cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 500)

#########V新增防止闖關V#########

model = load_model("./livenet/liveness.model")
le = pickle.loads(open("./livenet/le.pickle", "rb").read())
net = cv2.dnn.readNetFromCaffe("./livenet/detector/deploy.prototxt.txt", "./livenet/detector/res10_300x300_ssd_iter_140000.caffemodel")
label = ""
pred = ""

observed_resual=''
correct_count=5
def getLiveLabelfromImgandCoords(img, startX, startY, endX, endY, cw, ch):
    global model,le,net,label,pred
    lsy = startY
    lsx = startX
    ley = endY
    lex = endX
    fw = lex - lsx
    fh = ley - lsy
    rw = 1.3
    rh = 0
    if lsx - rw*fw > 0:
	    lsx = int(lsx - rw*fw)
    else:
	    lsx = 0
    if lsy - rh*fh > 0:
	    lsy = int(lsy - rh*fh)
    else:
	    lsy = 0
    if lex + rw*fw < cw:
	    lex = int(lex + rw*fw)
    else:
	    lex = cw
    if ley + rh*fh < ch:
	    ley = int(ley + rh*fh)
    else:
	    ley = ch
    liveFace = img[lsy:ley, lsx:lex]
    liveFace = cv2.resize(liveFace, (32, 32))

    liveFace = liveFace.astype("float") / 255.0
    liveFace = img_to_array(liveFace)
    liveFace = np.expand_dims(liveFace, axis=0)
    preds = model.predict(liveFace)[0]

    j = np.argmax(preds)

    label = le.classes_[j]
    pred = str(round(preds[j],2))

    if le.classes_[j] == "real" :
        if preds[j] > REAL_THRESHOLD:
            pass
        else:
            label = "false"

    return label
#########^0215新增防止闖關^#########  
import os
import time
def UnKnow_process(frame):
  Y=time.strftime("%Y", time.localtime()) 
  M=time.strftime("%m", time.localtime()) 
  D=time.strftime("%d", time.localtime()) 
  H=time.strftime("%H", time.localtime()) 
  Min=time.strftime("%M", time.localtime()) 
  Sec=time.strftime("%S", time.localtime()) 

  #a="{:}-{:}-{:} {:}:{:}:{:}".format(Y,M,D,H,Min,Sec)
  path="./UnKnow/{:}-{:}-{:}".format(Y,M,D)
  folder = os.path.exists(path)
  #判斷結果
  if not folder:
      #如果不存在，則建立新目錄
      os.makedirs(path)
      print('-----建立成功-----')

  savepath="./UnKnow/{:}-{:}-{:}/{:}h{:}m{:}s.jpg".format(Y,M,D,H,Min,Sec)
  cv2.imwrite(savepath, frame)
# 引入 requests 模組
import requests
IP="192.168.100.11"
def SendURL(sendword):
  #print(sendword)
  if sendword == "real":
    a="http://{:}/gpio/R_off".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/Y_off".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/G_on".format(IP)
    r = requests.get(a)
  elif sendword == "wait":
    a="http://{:}/gpio/R_off".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/Y_on".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/G_off".format(IP)
    r = requests.get(a)
  elif sendword == "false":
    a="http://{:}/gpio/R_on".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/Y_off".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/G_off".format(IP)
    r = requests.get(a)
  elif sendword == "UnKnow":
    a="http://{:}/gpio/R_on".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/Y_off".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/G_off".format(IP)
    r = requests.get(a)
    

  elif sendword == "off":
    a="http://{:}/gpio/R_off".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/Y_off".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/G_off".format(IP)
    r = requests.get(a)
    a="http://{:}/gpio/stay".format(IP)
    r = requests.get(a)

# 針對比對資料夾裡每張圖片做比對:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得描述子
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
  base = os.path.basename(f)
  # 依序取得圖片檔案人名
  candidate.append(os.path.splitext(base)[ 0])
  img = io.imread(f)

  # 1.人臉偵測
  dets = detector(img, 1)

  for k, d in enumerate(dets):
    # 2.特徵點偵測
    shape = sp(img, d)
 
    # 3.取得描述子，128維特徵向量
    face_descriptor = facerec.compute_face_descriptor(img, shape)

    # 轉換numpy array格式
    v = numpy.array(face_descriptor)
    descriptors.append(v)

#當攝影機打開時，對每個frame進行偵測
while(cap.isOpened()):
  
  #讀出frame資訊
  ret, frame = cap.read()
  h,w,l = np.shape(img)
  #print("h = {:}  w = {:}".format(h,w))

  #偵測人臉
  face_rects, scores, idx = detector.run(frame, 0)

  dist = []
  #print("len ace_rects = {:}".format(len(face_rects)))
  if len(face_rects)==0:SendURL("off")
  #if " " in face_rects : print("face_rects = None")
  for k, d in enumerate(face_rects):
    
    shape = sp(frame, d)
    face_descriptor = facerec.compute_face_descriptor(frame, shape)
    d_test = numpy.array(face_descriptor)

    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    
  
    # 計算歐式距離
    for i in descriptors:
      dist_ = numpy.linalg.norm(i -d_test)
      dist.append(dist_)

    # 將比對人名和比對出來的歐式距離組成一個dict
    c_d = dict( zip(candidate,dist))

    # 根據歐式距離由小到大排序
    cd_sorted = sorted(c_d.items(), key = lambda d:d[ 1])

    # 取得最短距離就為辨識出的人名
    if cd_sorted != [] :
      rec_name = cd_sorted[ 0][ 0]
    else : 
      rec_name="UnKnow"
      
    if rec_name!="UnKnow":
      observed_resual_singal=getLiveLabelfromImgandCoords(frame,x1,y1,x2,y2,w,h)
    else:
      observed_resual_singal="UnKnow"

    if observed_resual_singal == "real":
      std_correct_time+=1 
    else:                                
      std_correct_time=0
    #print("std_correct_time = {:}".format(std_correct_time))
 
    if abs(x1-x2)>100 and abs(y1-y2)>100 and observed_resual!="real":
      if std_correct_time>=correct_count:
        if observed_resual!="real":SendURL("real")
        observed_resual="real"
        color=(0,255,0)#green
        
      elif std_correct_time>0 and std_correct_time<correct_count:
        if observed_resual!="wait":SendURL("wait")
        observed_resual="wait"
        color=(255,128,0)#yellow
        
      elif std_correct_time==0 and rec_name!="UnKnow":
        if observed_resual!="false":SendURL("false")
        observed_resual="false"
        color=(255,0,0)#red

      elif rec_name=="UnKnow":
        if observed_resual!="UnKnow":
          SendURL("UnKnow") 
          UnKnow_process(frame)
        observed_resual="UnKnow"
        color=(0,0,255)#blue   

      # 將辨識出的人名印到圖片上面
      
      cv2.putText(frame, rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2, cv2. LINE_AA)
      # 以方框標示偵測的人臉
      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4, cv2. LINE_AA)      
    elif abs(x1-x2)<100 and abs(y1-y2)<100:
      #清除辨識結果
      observed_resual=""
      SendURL("off")
    

  frame = imutils.resize(frame, width = 600)
  frame = cv2.cvtColor(frame,cv2. COLOR_BGR2RGB)
  cv2.imshow( "Face Recognition", frame)
  #esc結束程式
  if cv2.waitKey( 10) == 27:
      break

cv2.destroyAllWindows()