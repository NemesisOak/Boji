import cv2 
import numpy as np #importado para as edições na imagem
import mediapipe as mp

#Captura e tratamento de imagem
camera = cv2.VideoCapture(0)
altura = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
largura = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))

while True: 

    #leitura de câmera   
    ret, frame = camera.read() 
    

    #espelha a imagem
    espelhada = np.flip(frame, axis=1)

    #mostra a imagem
    cv2.imshow('frame', espelhada)   
        
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break 

camera.release
cv2.destroyAllWindows()