from scipy.spatial import distance as dist
from operator import rshift
import cv2 as cv
import numpy as np
import mediapipe as mp
import mouse
import time
import pyautogui
import winsound
freq = 440  # Hz
#x=1919, y=1079
#50cm-35cm   x3<=xizq*(1.02)   x3>=xder*(0.98)   y3<=yarr*(1.02)     y3>=yaba*(0.97)
#20cm y3>=yaba*(0.93)   dif abajo: 26   dif arriba: 44
esc=0
circu=(960,500)
mov_x=700
mov_y=700
kaba=1
kaba2=1.03
karr=1
karr2=0.97
kizq=1
kizq2=0.97
kder=1
kder2=1.03
opab=0
zx=1
veloc1=5
veloc2=10
palabra2="1abaj 2arr 3izq 4der, 5veloc, 0salir"
palabra3="+ y - para ajustar"
 
font = cv.FONT_HERSHEY_PLAIN
 
mp_face_mesh = mp.solutions.face_mesh
EYE_AR_THRESH = 0.2  #margen para detectar si esta abierto o cerrado el ojo
EYE_AR_CONSEC_FRAMES = 3    #cantidad de frames para comparar entre el pesta inoluntario o voluntario
# contadores para los cierres de ojo izq y derecho
COUNTERI = 0
TOTALI = 0
COUNTERD =0
palabra="CENTRE SU ROSTRO CON LA CAMARA Y PULSE P"
 
cap = cv.VideoCapture(0)
#INICIO CALIBRACION
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
            cv.circle(framei, circu, 20, (255, 0, 0), -1)
            cv.imshow('imgCALIBRACION', framei)
            keyi = cv.waitKey(1)
            if keyi ==ord('p'):
                winsound.Beep(freq, 100)
                xi=mesh_pointsi[468][0]
                yi=mesh_pointsi[468][1]
                palabra="IZQUIERDA Y APRETAR a"
                circu=(420,500)
            elif keyi == ord('a'):
                winsound.Beep(freq, 100)
                xizq=mesh_pointsi[468][0]
                yizq=mesh_pointsi[468][1]
                palabra="DERECHA Y APRETAR d"
                circu=(1500,500)
            elif keyi ==ord('d'):
                winsound.Beep(freq, 100)
                xder=mesh_pointsi[468][0]
                yder=mesh_pointsi[468][1]
                palabra="ARRIBA Y APRETAR w"
                circu=(960,20)
            elif keyi ==ord('w'):
                winsound.Beep(freq, 100)
                xarr=mesh_pointsi[468][0]
                yarr=mesh_pointsi[468][1]
                palabra="ABAJO Y APRETAR s"
                circu=(960,1000)
            elif keyi ==ord('s'):
                winsound.Beep(freq, 100)
                xaba=mesh_pointsi[468][0]
                yaba=mesh_pointsi[468][1]
                palabra="q PARA SALIR"
            elif keyi ==ord('q'):
                esc=1


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
            
            if eari<EYE_AR_THRESH:  
                    COUNTERI += 1
                    if(COUNTERI==10):
                        winsound.Beep(329, 500)
                    if(COUNTERI==40):
                        winsound.Beep(freq, 500)
            
            else: 
                if(COUNTERI>=EYE_AR_CONSEC_FRAMES):
                    if(COUNTERI<10):
                        cv.putText(frame, "CLICKKKKK", (50, 150), font, 7, (255, 0, 0))
                        mouse.click('left')
                        print("CLICK")
                        COUNTERI = 0
                    elif(COUNTERI>=10 and COUNTERI<40):
                        print("STOP")
                        COUNTERI = 0
                        zx=zx*-1
                    else:
                        mov_x=1000
                        mov_y=500
                        mouse.move(1000,500)
                        print("centro")
                        COUNTERI = 0
            
                elif(zx==1): #SI ENTRO ACA TENGO LOS DOS OJOS ABIERTOS, HAGO EL SEGUIMIENTO
                    COUNTERI = 0
                    x3=mesh_points[468][0]
                    y3=mesh_points[468][1]
                    if(x3<=xizq*(kizq)):
                        if(mov_x>0):
                            cv.putText(frame,"IZQUIERDA",(350,200),font,2,(0,0,255),3)
                            mov_x=mov_x-veloc1
                            if(x3<=(xizq*kizq2)):
                                mov_x=mov_x-veloc2
                            mouse.move(mov_x,mov_y)
                    elif(x3>=xder*(kder)):
                        if(mov_x<1919):
                            cv.putText(frame,"DERECHA",(350,200),font,2,(0,0,255),3)
                            mov_x=mov_x+veloc1
                            if(x3>=(xder*kder2)):
                                mov_x=mov_x+veloc2
                            mouse.move(mov_x,mov_y)
                    else:
                        cv.putText(frame,"CENTRO",(350,200),font,2,(0,0,255),3)
            
                    if(y3<=yarr*(karr)):
                        if(mov_y>0):
                            cv.putText(frame,"ARRIBA",(50,200),font,2,(0,0,255),3)
                            mov_y=mov_y-veloc1
                            if(y3<=(yarr*karr2)):
                                mov_y=mov_y-veloc2
                            mouse.move(mov_x,mov_y)
                    elif(y3>=yaba*(kaba)):
                        cv.putText(frame,"ABAJO",(50,200),font,2,(0,0,255),3)
                        if(mov_y<1079):
                            mov_y=mov_y+veloc1
                            if(y3>=(yarr*kaba2)):
                                mov_y=mov_y+veloc2
                            mouse.move(mov_x,mov_y)
                    else:
                        cv.putText(frame,"CENTRO",(50,200),font,2,(0,0,255),3)
                    cv.putText(frame,palabra2,(50,800),font,2,(0,0,255),3)
                    cv.putText(frame,palabra3,(50,900),font,2,(0,0,255),3)
                    key = cv.waitKey(1)
                    if key ==ord('x'):
                        break
                    elif key ==ord('1'): #ajusta abajo
                        opab=1
                        palabra2="AJUSTE ABAJO"
                    elif key ==ord('2'): #ajusta arriba
                        opab=2
                        palabra2="AJUSTE ARRIBA"
                    elif key ==ord('3'): #ajusta izquierda
                        opab=3
                        palabra2="AJUSTE IZQUIERDA"
                    elif key ==ord('4'): #ajusta derecha
                        opab=4
                        palabra2="AJUSTE DERECHA"
                    elif key ==ord('5'): #ajusta velocidad
                        opab=5
                        palabra2="AJUSTE VELOCIDAD"
                    elif key ==ord('0'):    #SALIR DE AJUSTE
                        opab=0
                        palabra2="1abaj 2arr 3izq 4der, 5veloc, 0salir"
                    elif key ==ord('-') and opab==1:
                        kaba=kaba+0.005
                        kaba2=kaba2+0.01
                    elif key ==ord('+') and opab==1:
                        kaba=kaba-0.005
                        kaba2=kaba2-0.005
                    elif key ==ord('+') and opab==2:
                        karr=karr+0.005
                        karr2=karr2+0.005
                    elif key ==ord('-') and opab==2:
                        karr=karr-0.005
                        karr2=karr2-0.005
                    elif key ==ord('+') and opab==3:
                        kizq=kizq+0.005
                        kizq2=kizq2+0.005
                    elif key ==ord('-') and opab==3:
                        kizq=kizq-0.005
                        kizq2=kizq2-0.005
                    elif key ==ord('-') and opab==4:
                        kder=kder+0.005
                        kder2=kder2+0.005
                    elif key ==ord('+') and opab==4:
                        kder=kder-0.005
                        kder2=kder2-0.005
                    elif key ==ord('-') and opab==5:
                        veloc1=veloc1-1
                        veloc2=veloc2-1
                    elif key ==ord('+') and opab==5:
                        veloc1=veloc1+1
                        veloc2=veloc2+1

                            
        cv.imshow('img', frame)
cap.release()
#cv.destroyAllWindows()