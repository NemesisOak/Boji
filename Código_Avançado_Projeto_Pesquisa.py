#!/usr/bin/env python

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

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True : 
        #leitura de câmera   
        ret, frame = camera.read() 

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
                
        #espelha a imagem
        espelhada = np.flip(frame, axis=1)

        #mostra a imagem
        cv2.imshow('frame', espelhada)   
        
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break 

    camera.release
    cv2.destroyAllWindows()
    