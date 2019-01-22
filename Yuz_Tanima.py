import numpy as np
import cv2
import os
'''y�z tan�ma i�in gerekli k�t�phaneler import edildi.
    cv2--> g�rsel veri i�leme i�in kullan�lan k�t�phane
    numpy--> say�sal veri i�leme i�in kullan�lan k�t�phane
'''

list = os.listdir("C:/Users/Alkan/PycharmProjects/Yuz_Tanima")

for item in list:
    if item.endswith(".jpg"):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        '''Cascade xml dosyalar� olup, haz�r dosyalard�r.
                Face cascade --> y�z hatlar�n�n alg�lanmas�n� sa�layan bir xml dosyas�d�r.
                Eye cascade --> g�z alg�lamak i�i kullan�lan bir xml dosyas�d�r.
            '''
        img = cv2.imread(str(item))
        ''' resmimiz cv2 k�t�phanesi yard�m�yla klas�rden okunuyor. '''

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        '''resim okunduktan sonra 3 skaladan olu�an resmin cv2 yard�m�yla 
                gri skalas� di�er sklalardan ayr��t�r�l�yor
            '''
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        ''' detectMutliScale --> farkl� boyuttaki nesneleri alg�lamay� sa�layan CascadeClassifier fonksiyonudur. 
            '''
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        """ Bu k�s�mda alg�lanan y�z ve g�z�n etraf�na �er�eve �iziyor"""
        cv2.imshow('img', img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()