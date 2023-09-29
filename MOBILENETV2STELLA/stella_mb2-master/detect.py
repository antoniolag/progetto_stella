import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import os
import time 
from bs4 import BeautifulSoup


inizio = time.time()
configs = config_util.get_configs_from_pipeline_file("network/ssd/pipeline.config") 
model_config = configs['model'] # recreating model config
detection_model = model_builder.build(model_config=model_config, is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join("train/", 'ckpt-23')).expect_partial()

path2label_map = 'data/dataset/labels_map.pbtxt' 
category_index = label_map_util.create_category_index_from_labelmap(path2label_map,use_display_name=True)

print("Modello caricato in " + str(time.time() - inizio ) + " secondi")

def detect_fn(image):


    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

chiave = 0
dizionario_classe = {}
with open("stella_dataset/labels.txt") as f:
    lista_classi = f.readlines()
for i in range(len(lista_classi)):
    lista_classi[i] = lista_classi[i].replace("\n","")
for classe in lista_classi:
    nome_classe = classe.replace("\n","")
    dizionario_classe[chiave] = nome_classe
    chiave +=1
print(dizionario_classe)

tutte = 0
giuste = 0
sbagliate = 0
veri =[]
predetti = []

for file in os.listdir("data/dataset/test/"):
    inizio = time.time()
    
    if file.endswith(".jpg"):
        xml_path = file.replace(".jpg",".xml")
        with open("data/dataset/test/" + xml_path, 'r') as f:
            data = f.read()
        bs_data = BeautifulSoup(data, 'xml')
        for tag in bs_data.find_all('name'):
            vero = (str(tag).split(">")[1].split("<")[0])
        tutte += 1
        image = cv2.imread("data/dataset/test/" + file)
        image_h,image_w ,_= image.shape
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']
        
        detections = {key: value for key, value in detections.items() if key in key_of_interest}

        box = ((detections['detection_boxes'])[0])
        if(detections['detection_scores'][0]> 0.7):
            y1abs, x1abs = int(box[0] * image_h), int(box[1] * image_w)
            y2abs, x2abs = int(box[2] * image_h), int(box[3] * image_w)
            cv2.rectangle(image, (x1abs, y1abs), (x2abs,y2abs), (10, 255, 0), 2)
            cv2.putText(image, dizionario_classe[detections['detection_classes'][0]], (x1abs, y1abs), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text<
            cv2.imwrite("data/dataset/test/out/" + file ,image)
                        
            if (dizionario_classe[detections['detection_classes'][0]] == vero):
                giuste +=1
                veri.append(vero)
                predetti.append(dizionario_classe[detections['detection_classes'][0]])
            else:
                
                veri.append(vero)
                predetti.append(dizionario_classe[detections['detection_classes'][0]])
                sbagliate +=1
        else:
            print("siamo qui ")
            print(vero)
            print(dizionario_classe[detections['detection_classes'][0]])
            sbagliate +=1
            veri.append(vero)
            predetti.append(dizionario_classe[detections['detection_classes'][0]])

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
cm = (confusion_matrix(veri, predetti, labels=lista_classi))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Aggiungi etichette agli assi
tick_marks = np.arange(len(lista_classi))
plt.xticks(tick_marks, lista_classi, rotation=45)
plt.yticks(tick_marks, lista_classi)

# Aggiungi i valori numerici nella matrice
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

# Aggiungi etichette e titolo
plt.ylabel('Etichetta reale')
plt.xlabel('Etichetta predetta')
plt.title('Matrice di confusione')

# Salva l'immagine
plt.savefig('matrice_confusione.png')
print(veri)
print(predetti)
print(giuste)
print(sbagliate)
print((giuste+sbagliate) == tutte)
print(tutte)
print((giuste *100)/ tutte )




from sklearn.metrics import precision_score, recall_score

# Supponiamo che 'y_true' sia la lista delle etichette di classe reali e 'y_pred' sia la lista delle previsioni del tuo modello.
# Queste liste dovrebbero avere lo stesso numero di elementi.

# Calcolo della precisione
precision = precision_score(veri, predetti, average='weighted')  # Puoi scegliere 'macro', 'micro', o 'weighted'.

# Calcolo del richiamo
recall = recall_score(veri, predetti, average='weighted')  # Puoi scegliere 'macro', 'micro', o 'weighted'.

print(f'Precision: {precision}')
print(f'Recall: {recall}')









from sklearn.metrics import f1_score

# Supponiamo che 'y_true' sia la lista delle etichette di classe reali e 'y_pred' sia la lista delle previsioni del tuo modello.
# Queste liste dovrebbero avere lo stesso numero di elementi.

# Calcolo del punteggio F1
f1 = f1_score(veri, predetti, average='weighted')  # Puoi scegliere 'macro', 'micro', o 'weighted'.

print(f'F1-Score: {f1}')

