from scipy.spatial import distance as dist
from operator import rshift
import cv2 as cv
import numpy as np
import mediapipe as mp
import mouse
import time
import pyautogui
#x=1919, y=1079

#defino funcion calibracion 
def calibracion():
    esc=0
    circle=((pyautogui.size()[0]//2),(pyautogui.size()[1]//2))
    palabra="Mire el punto central y PULSE p"
    font = cv.FONT_HERSHEY_PLAIN
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    ) as face_mesh:
        while esc==0:
            reti, framei = cap.read() 
            if not reti:
                break
            framei = cv.flip(framei, 1)  #doy vuelta la imagen para que este alineada a la persona
            framei = cv.resize(framei, (1920, 1080))
            rgb_framei = cv.cvtColor(framei, cv.COLOR_BGR2RGB)
            img_hi, img_wi = framei.shape[:2]
            resultsi = face_mesh.process(rgb_framei)
            if resultsi.multi_face_landmarks:
                mesh_pointsi=np.array([np.multiply([p.x, p.y], [img_wi, img_hi]).astype(int) for p in resultsi.multi_face_landmarks[0].landmark])
                cv.putText(framei,palabra,(200,200),font,2,(0,0,255),3)
                cv.circle(framei, circle, 20, (255, 0, 0), -1)
                cv.imshow('imgCALIBRACION', framei)
                keyi = cv.waitKey(1)
                if keyi ==ord('p'):
                    xic=mesh_pointsi[468][0]
                    yic=mesh_pointsi[468][1]
                    palabra="MIRE EL PUNTO DE LA IZQUIERDA DE LA PANTALLA Y APRETAR q"
                    circle=(0,(pyautogui.size()[1]//2))
                elif keyi ==ord('q'):
                    xizqc=mesh_pointsi[468][0]                
                    palabra="MIRE EL PUNTO DE LA DERECHA DE LA PANTALLA Y APRETAR w"
                    circle=(pyautogui.size()[0]-10,(pyautogui.size()[1]//2))
                elif keyi ==ord('w'):
                    xderc=mesh_pointsi[468][0]                
                    palabra="MIRE EL PUNTO ARRIBA DE LA PANTALLA Y APRETAR e"
                    circle=((pyautogui.size()[0]//2),0)
                elif keyi ==ord('e'):
                    yarrc=mesh_pointsi[468][1]
                    palabra="MIRE ABAJO DE LA PANTALLA Y APRETAR r"
                    circle=((pyautogui.size()[0]//2),pyautogui.size()[1]-10)
                elif keyi ==ord('r'):
                    yabac=mesh_pointsi[468][1]
                    esc=1
    return xic,yic,xizqc,xderc,yarrc,yabac

font = cv.FONT_HERSHEY_PLAIN
mp_face_mesh = mp.solutions.face_mesh
#puntos del contorno de ojos e iris de ambos ojos
#LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
#RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
#LEFT_IRIS = [474,475, 476, 477]
#RIGHT_IRIS = [469, 470, 471, 472]
EYE_AR_THRESH = 0.2  #margen para detectar si esta abierto o cerrado el ojo
EYE_AR_CONSEC_FRAMES = 3    #cantidad de frames para comparar entre el pesta inoluntario o voluntario
# contadores para los cierres de ojo izq y derecho
COUNTERI = 0
TOTALI = 0
COUNTERD =0
TOTALD= 0
x32=0
y32=0
 
cap = cv.VideoCapture(0)

pyautogui.moveTo(pyautogui.size()[0]//2,pyautogui.size()[1]//2,duration=0.25)  #centro de la pantalla
mov_x,mov_y=pyautogui.position()  #centro de la pantalla
centrox=pyautogui.size()[0]//2
centroy=pyautogui.size()[1]//2
[xi,yi,xizq,xder,yarr,yaba]=calibracion()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
            
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)  #doy vuelta la imagen para que este alineada a la persona
        frame = cv.resize(frame, (1920, 1080))
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            b = mesh_points[468]  #CENTRO DEL OJO izq
            cv.circle(frame, b, 3, (255, 0, 0), -1)
            c = mesh_points[6]   #PUNTO CENTRO ENTRE LOS OJOS
            cv.circle(frame, c, 3, (255, 0, 0), -1)
            d = mesh_points[159]        #alto de ojo izq
            cv.circle(frame, d, 3, (255, 0, 0), -1)
            dd = mesh_points[386]        #alto de ojo derecha
            cv.circle(frame, dd, 3, (255, 0, 0), -1)
            e = mesh_points[145]        #bajo de ojo iquierdo
            cv.circle(frame, e, 3, (255, 0, 0), -1)
            ed = mesh_points[374]        #bajo de ojo derecha
            cv.circle(frame, ed, 3, (255, 0, 0), -1)
            f = mesh_points[33]        #izquierda de ojo izquierdo
            cv.circle(frame, f, 3, (255, 0, 0), -1)
            fd = mesh_points[263]        #izquierda de ojo derecha
            cv.circle(frame,fd, 3, (255, 0, 0), -1)
            g = mesh_points[133]        #derecha de ojo izquierdo
            cv.circle(frame, g, 3, (255, 0, 0), -1)
            gd = mesh_points[362]        #derecha de ojo derecha
            cv.circle(frame, gd, 3, (255, 0, 0), -1)
                                    
            #calculo de la relacion de distancias del ojo
            #ear=d-e/f-g si ear=0 parpadeo
            eari=dist.euclidean(d, e)/dist.euclidean(g, f)
            eard=dist.euclidean(dd, ed)/dist.euclidean(gd, fd)
            if (xi-xizq!=0):
                Gxi=centrox//(xi-xizq)
            if (xi-xder!=0):    
                Gxd=(centrox-pyautogui.size()[0])//(xi-xder)  #positivo
            if (yi-yaba!=0):
                Gya=(centroy-pyautogui.size()[1])//(yi-yaba)  #positivo
            if (yi-yarr!=0):
                Gyar=centroy//(yi-yarr)
            
            x1p=0
            
            #saco un promedio de la ubicacion del punto central del ojo
            
            x1=mesh_points[468][0]
            x1p=x1
            
            
            
            y1p=0
            
            y1=mesh_points[468][1]
            y1p=y1
            
            
            y3=yi-y1p
            x3=xi-x1p
            #aca comienza el programa, y comienza analizando la apertura de los ojos, dependeiendo de ese dato toma una accion
            #comparo ear con la referencia

        

            if eari<EYE_AR_THRESH:  
                COUNTERI += 1
                if COUNTERI >= EYE_AR_CONSEC_FRAMES:    #SI SUPERO LOS FRAMES HAGO CLICK SINO SIGO EL SEGUIMIENTO
                    TOTALI += 1
                    pyautogui.click(x=mov_x, y=mov_y)  #click en la posicion actual
                    # reseteo el contador de frames
                    COUNTERI = 0
            
            else: #SI ENTRO ACA TENGO LOS DOS OJOS ABIERTOS, HAGO EL SEGUIMIENTO
            
                #y2=float(mesh_points[133][1]) #comparo con la relacion entre el bodr cerca de la nariz y el centro del ojo

                if (y3>0.5 and x3>0.5):
                    cv.putText(frame,"ARRIBA e IZQUIERDA",(50,200),font,2,(0,0,255),3)
                    if(mov_y>0 and mov_x>0):
                        if((y32-1<=y3<y32+0.5) and (x32-1<=x3<x32+0.5)):
                            mov_x,mov_y=mouse.get_position()
                            mouse.move(mov_x,mov_y)
                        else:
                            mov_y=yi-y3*Gyar
                            mov_x=xi-x3*Gxi
                            mouse.move(mov_x,mov_y)
                elif (y3>0.5 and x3<-0.5) :
                    cv.putText(frame,"ARRIBA Y DERECHA",(50,200),font,2,(0,0,255),3)
                    if(mov_y>0 and mov_x<pyautogui.size()[0]):
                        if((y32-1<=y3<y32+0.5) and (x32-1<=x3<x32+0.5)):
                            mov_x,mov_y=mouse.get_position()
                            mouse.move(mov_x,mov_y)
                        else:
                            mov_y=yi-y3*Gya
                            mov_x=xi-x3*Gxd
                            mouse.move(mov_x,mov_y)
                elif (y3<-0.5 and x3>0.5):
                    cv.putText(frame,"ABAJO IZQUIERDA",(50,200),font,2,(0,0,255),3)
                    if(mov_y<pyautogui.size()[1] and mov_x>0):
                        if((y32-1<=y3<y32+0.5) and (x32-1<=x3<x32+0.5)):
                            mov_x,mov_y=mouse.get_position()
                            mouse.move( mov_x,mov_y)
                        else:
                            mov_y=yi-y3*Gya
                            mov_x=xi-x3*Gxi
                            mouse.move(mov_x,mov_y)
                elif (y3<-0.5 and x3<-0.5) :
                    cv.putText(frame,"ABAJO Y derecha",(50,200),font,2,(0,0,255),3)
                    if(mov_y<pyautogui.size()[1] and mov_x<pyautogui.size()[0]):
                        if((y32-1<=y3<y32+0.5) and (x32-1<=x3<x32+0.5)):
                            mov_x,mov_y=mouse.get_position()
                            mouse.move( mov_x,mov_y)
                        else:
                            mov_y=yi-y3*Gya
                            mov_x=xi-x3*Gxd
                            mouse.move(mov_x,mov_y)
                else :
                    cv.putText(frame,"CENTRO",(50,200),font,2,(0,0,255),3)
                    if((y32-1<=y3<y32+0.5) and (x32-1<=x3<x32+0.5)):
                        mov_x,mov_y=mouse.get_position()
                        mouse.move( mov_x,mov_y)
                    else:
                        mov_y=yi-y3*Gya
                        mov_x=xi-x3*Gxi
                        mouse.move(mov_x,mov_y)
            
            if eard<EYE_AR_THRESH:
                COUNTERD += 1
                if (COUNTERD >= EYE_AR_CONSEC_FRAMES):
                    TOTALD += 1
                    cv.putText(frame, "STOP", (50, 150), font, 7, (255, 0, 0))
                    mov_x,mov_y=pyautogui.position()
                    mouse.move(mov_x,mov_y)
                    COUNTERD = 0                 
            keyc = cv.waitKey(1)
            if keyc ==ord('c'):
                while True:
                    palabra="Calibracion punto de referencia, oprima c para confirmar"
                    cv.putText(frame,palabra,(100,100),font,2,(0,0,255),3)
                    cv.imshow('img', frame)
                    
                    keycc = cv.waitKey(1)
                    if keycc ==ord('c'):
                        xi=mesh_points[468][0]
                        yi=mesh_points[468][1]
                        break


            cv.putText(frame,str(x32),(100,100),font,2,(0,0,255),3)
            cv.putText(frame,str(x3),(100,150),font,2,(0,0,255),3)
            cv.putText(frame,str(y32),(150,250),font,2,(0,0,255),3)
            cv.putText(frame,str(y3),(150,300),font,2,(0,0,255),3)
            y32=y3
            x32=x3


        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
        


cap.release()
cv.destroyAllWindows()
 