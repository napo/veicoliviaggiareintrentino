#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import requests
import torch
#from ultralytics.nn.tasks import attempt_load_one_weight
#import locale
from ultralytics import YOLO
import warnings
import os
import cv2
import numpy as np

import areas.areas as ar
# workaround https://github.com/pytorch/vision/issues/4156
# torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
#model = torch.hub.load('ultralytics/yolov8', 'yolov8n')  # or yolov5m, yolov5l, yolov5x, custom
model = YOLO("yolov8n.pt")
warnings.filterwarnings('ignore')
#locale.setlocale(locale.LC_TIME, 'it_IT.utf8')
webcam_json_url = 'https://vit.trilogis.it/json/webcam'
output_data = "data" + os.sep + "vehicles_on_trentino_webcams.parquet"
last_output_data = "data" + os.sep + "last_vehicles_on_trentino_webcams.csv"


# In[61]:


webcams = pd.DataFrame(requests.get(webcam_json_url,verify=False).json()['webcams']['webcam'])


# In[62]:


webcams['Id'] = webcams['Id'].astype(int)
webcams['Cod'] = webcams['Cod'].astype(int)
webcams['Nome'] = webcams['Nome'].astype(str)
webcams['Direzione'] = webcams['Direzione'].astype(str)
webcams['Url_Immagine'] = webcams['Url_Immagine'].astype(str)
webcams['IP_Webcam'] = webcams['IP_Webcam'].astype(str)
webcams['Km'] = webcams['Km'].astype(str)
webcams['Strada'] = webcams['Strada'].astype(str)
webcams['Localita'] = webcams['Localita'].astype(str)
webcams['ZonaTN'] = webcams['ZonaTN'].astype(int)
webcams['Lat'] = webcams['Lat'].apply(lambda x: float(x) if x!='' else 0)
webcams['Lng'] = webcams['Lng'].apply(lambda x: float(x) if x!='' else 0)
webcams['Monitoraggio'] = webcams['Monitoraggio'].astype(bool)
webcams['Live'] = webcams['Live'].astype(bool)
webcams['TS_Image'] = webcams['TS_Image'].astype(str)


# In[63]:


#rimozione webcam con webcam dismesse
webcams = webcams[webcams['Url_Immagine'] != 'http://vit.trilogis.it/cam/webcam_outdated.jpg']


# In[64]:


def to_timestamp(x):
    mesi = {
        'gennaio': '01',
        'febbraio': '02',
        'marzo': '03',
        'aprile': '04',
        'maggio': '05',
        'giugno': '06',
        'luglio': '07',
        'agosto': '08',
        'settembre': '09',
        'ottobre': '10',
        'novembre': '11',
        'dicembre': "12"
    }
    data = x.split(",")[1].split(" ")
    day = data[1]
    month = mesi[data[2]]
    year = data[3]
    hour = data[4]
    rtime = year + "-" + month + "-" + day + " " + hour + ",0"
    return rtime


# In[65]:


webcams['timestamp'] = webcams['TS_Image'].apply(lambda x: to_timestamp(x))
webcams['timestamp'] = pd.to_datetime(webcams.timestamp, format='%Y-%m-%d %H:%M:%S,%f')
vehicles = ['car','truck','bus','train','motorcycle']
required_conf = {'car': 0.3, 'truck': 0.3, 'bus': 0.3, 'train': 0.7, 'motorcycle': 0.3}

# COCO class list (https://github.com/pjreddie/darknet/blob/master/data/coco.names)
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# In[66]:


# 45 -> train 0.50 (è una staccionata)
# 34 -> train 0.55 (sono in guardrail)
# CAM65 -> auto in parcheggio (eliminare pixel?)    FIXED
# CAM100 -> auto in parcheggio (eliminare pixel?)   FIXED
# CAM111 -> auto in parcheggio (eliminare pixel?)   FIXED
# CAM125 -> auto in parcheggio (eliminare pixel?)   FIXED
# CAM126 -> auto in parcheggio (eliminare pixel?)   FIXED


# In[67]:


def identifyVehicles(id,indf):
    num_vehicles = 0
    indf = indf[indf['Id'] == id]
    url = indf['Url_Immagine'].values[0]
    
    try:
        if (url.find("webcam_outdated.jpg") == -1):
            try:
                results = model.predict([url])
                
                area_name = url.split("/")[4]
                
                area = ar.area[f"{area_name}"]

                # to save the image with the bounding boxes decoment the following 5 lines:
                # response = requests.get(url)
                # with open(f'docs/results/{area_name}', 'wb') as file:
                #     file.write(response.content)
                # frame = cv2.imread(f'docs/results/{area_name}')
                # cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,0),2)
                
                a = results[0].boxes.data
                px = pd.DataFrame(a.cpu().numpy()).astype('float')
                
                for index, row in px.iterrows(): 
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    conf = float(row[4])
                    class_list_id = int(row[5])
                    name = class_list[class_list_id]

                    if any(vehicle in name for vehicle in vehicles) and conf >= required_conf[name]:
                        cx = int(x1 + x2) // 2 
                        cy = int(y1 + y2) // 2 
                        r = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
                        
                        # if r = 1 -> inside the are
                        # if r = -1 -> outside the area
                        if r == 1:

                            # to save the image with the bounding boxes decoment the following 5 lines:
                            # data = f'{name} {conf:.2f}'
                            # (w, h), _ = cv2.getTextSize(data, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                            # cv2.rectangle(frame,(x1,y1),(x2,y2),(35,110,255),2)
                            # cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (35,110,255), -1)
                            # cv2.putText(frame,str(data),(x1, y1-5),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,255),1)

                            num_vehicles += 1

                # to save the image with the bounding boxes decoment the following line:
                # cv2.imwrite(f'docs/results/{area_name}', frame)            

            except Exception as ex:
                pass
    except OSError as e:
        pass

    return(num_vehicles) #['class'].sum())


webcams['veicoli'] = webcams['Id'].apply(lambda x: identifyVehicles(x,webcams))



columns = {
    'Id':'id','Cod':'codice',"Nome":'nome',
    'Direzione':'direzione','Url_Immagine':'url',
    'Attiva': 'attiva',
    'Comune':'comune','Comunita':'comunita_valle',
    'IP_Webcam':'ip_webcam','Km':'km','Strada':'strada',
    'Localita':'localita','ZonaTN':'zona_tn', 'Lat':'latitudine',
    'Lng':'longitude','Monitoraggio':'monitoraggio',
    'Live':'live','TS_Image':'data'}


# In[70]:


webcams.rename(columns=columns, inplace=True)


# In[71]:


del webcams['data']
del webcams['ip_webcam']
del webcams['monitoraggio']
del webcams['live']
del webcams['attiva']


# In[72]:


if os.path.exists(output_data):
    last_out = pd.read_parquet(output_data)
    last_timestamp = last_out.timestamp.max()
    row = webcams.shape[0]
    if row !=0:
        actual_timestamp = webcams.timestamp.max().strftime("%Y-%m-%d %H:%M:%S")
        if str(last_timestamp) != str(actual_timestamp):
            newdata = pd.concat([last_out, webcams]) 
            newdata['timestamp'] = pd.to_datetime(newdata.timestamp, format='%Y-%m-%d %H:%M:%S')
            newdata.to_parquet(output_data, index=False)
else:
    webcams.to_parquet(output_data, index=False)


# In[73]:


#webcams.timestamp.max().strftime("%Y-%m-%d %H:%M:%S")


# In[74]:


webcams.to_csv(last_output_data, index=False)

