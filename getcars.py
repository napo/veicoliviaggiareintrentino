#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import requests
import torch
from ultralytics.nn.tasks import attempt_load_one_weight
#import locale
import warnings
import os
# workaround https://github.com/pytorch/vision/issues/4156
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # or yolov5m, yolov5l, yolov5x, custom
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


# In[66]:


# 45 -> train 0.50 (è una staccionata)
# 34 -> train 0.55 (sono in guardrail)
# CAM65 -> auto in parcheggio (eliminare pixel?)
# CAM100 -> auto in parcheggio (eliminare pixel?)
# CAM111 -> auto in parcheggio (eliminare pixel?)
# CAM125 -> auto in parcheggio (eliminare pixel?)
# CAM126 -> auto in parcheggio (eliminare pixel?)
ids_clean = [34]


# In[67]:


def identifyVehicles(id,indf):
    num_vehicles = 0
    indf = indf[indf['Id'] == id]
    url = indf['Url_Immagine'].values[0]
    try:
        if (url.find("webcam_outdated.jpg") == -1):
            try:
                results = model([url])
                #results.save("docs" + os.sep + "results")
                if (len(results) >= 1):
                    results_df = results.pandas().xyxy[0]
                    results_df = results_df[results_df['confidence'] >= 0.4]
                    results_df = results_df[results_df['name'].isin(vehicles)]
                    num_vehicles = results_df.shape[0] 
                    if id in ids_clean:
                        num_vehicles = num_vehicles - 1
                        if num_vehicles < 0:
                            num_vehicles = 0
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

