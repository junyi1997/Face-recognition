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
#選擇第一隻攝影機
cap = cv2.VideoCapture(0)
#調整預設影像大小，預設值很大，很吃效能
cap.set(cv2. CAP_PROP_FRAME_WIDTH, 650)
cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 500)

# 比對人臉圖片資料夾名稱
faces_folder_path = "./rec"

#取得預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
#根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat')
# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表
candidate = []

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
    dets = detector(img, 0)
    for k, d in enumerate(dets):
# 2.特徵點偵測
        shape = predictor(img, d)
# 3.取得描述子，128維特徵向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)    
# 轉換numpy array格式
        v = numpy.array(face_descriptor)
        descriptors.append(v)

  #當攝影機打開時，對每個frame進行偵測
while(cap.isOpened()):
    #讀出frame資訊
    ret, frame = cap.read()

   

    #找出特徵點位置
    shape = predictor(frame, d)
    dets,_,_ = detector.run(frame, 0)

    dist = []
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        d_test = numpy.array(face_descriptor)
        
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        #以方框標示偵測的人臉
        cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
        
  # 計算歐式距離
        for i in descriptors:
            dist_ = numpy.linalg.norm(i -d_test)
            dist.append(dist_)

    # 將比對人名和比對出來的歐式距離組成一個dict
        c_d = dict( zip(candidate,dist))
        print(c_d)
# 根據歐式距離由小到大排序
        cd_sorted = sorted(c_d.items(), key = lambda d:d[ 1])
# 取得最短距離就為辨識出的人名
        rec_name = cd_sorted[ 0][ 0]
        
# 將辨識出的人名印到圖片上面
        cv2.putText(frame, rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2, cv2. LINE_AA)



      #繪製68個特徵點
        #for i in range( 81):
        #    cv2.circle(frame,(shape.part(i).x,shape.part(i).y), 3,( 0, 0, 255), 2)
        #    cv2.putText(frame, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
    #輸出到畫面
    cv2.imshow( "Face Detection", frame)

    #如果按下ESC键，就退出
    if cv2.waitKey( 10) == 27:
        break
#釋放記憶體
cap.release()
#關閉所有視窗
cv2.destroyAllWindows()