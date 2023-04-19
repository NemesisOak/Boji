#!/usr/bin/env python
from imp import load_module
import tensorflow as tf
from tensorflow import keras
#from keras import load_model
import cv2 
import numpy as np #importado para as edições na imagem
import mediapipe as mp

#Captura e tratamento de imagem
camera = cv2.VideoCapture(0)
altura = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
largura = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))

#Função mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#Keras 
model = keras.models.load_model(r"C:\Users\lauzi\OneDrive\Documents\Projeto de Pesquisa 2022\Teachable Machine\keras_model.h5")
np.set_printoptions(suppress=True)
model = load_model("keras_Model.h5", compile=False) #Este é o modelo que será importado
class_names = open("labels.txt", "r").readlines()


with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True : 
        #leitura de câmera   
        ret, frame = camera.read() 

        #espelha a imagem
        espelhada = np.flip(frame, axis=1)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)  
        
        #edição mediapipe
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        #analise keras
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1    
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        #mostra a imagem
        cv2.imshow('frame', espelhada)   

        #previsão resultado
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break 

    camera.release
    cv2.destroyAllWindows()
    
