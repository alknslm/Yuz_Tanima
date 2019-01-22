import numpy as np
import cv2
import os
'''yüz tanýma için gerekli kütüphaneler import edildi.
    cv2--> görsel veri iþleme için kullanýlan kütüphane
    numpy--> sayýsal veri iþleme için kullanýlan kütüphane
'''

list = os.listdir("C:/Users/Alkan/PycharmProjects/Yuz_Tanima")

for item in list:
    if item.endswith(".jpg"):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        '''Cascade xml dosyalarý olup, hazýr dosyalardýr.
                Face cascade --> yüz hatlarýnýn algýlanmasýný saðlayan bir xml dosyasýdýr.
                Eye cascade --> göz algýlamak içi kullanýlan bir xml dosyasýdýr.
            '''
        img = cv2.imread(str(item))
        ''' resmimiz cv2 kütüphanesi yardýmýyla klasörden okunuyor. '''

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        '''resim okunduktan sonra 3 skaladan oluþan resmin cv2 yardýmýyla 
                gri skalasý diðer sklalardan ayrýþtýrýlýyor
            '''
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        ''' detectMutliScale --> farklý boyuttaki nesneleri algýlamayý saðlayan CascadeClassifier fonksiyonudur. 
            '''
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        """ Bu kýsýmda algýlanan yüz ve gözün etrafýna çerçeve çiziyor"""
        cv2.imshow('img', img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()