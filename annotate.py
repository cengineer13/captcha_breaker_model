from imutils import paths
from imutils import contours
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of images")
ap.add_argument("-a", "--annotation", required=True,
                help="Path to output directory of annotation")
args = vars(ap.parse_args())

#image_paths = list(paths.list_images(args["input"]))
image_paths = list(paths.list_images('downloads/'))
counts = {}

#image path boylab loop qilamiz.
for (i, image_path) in enumerate(image_paths[:1]):
    print(f"INFO Processing images {i+1} / {len(image_paths)}")

    try:
        image = cv2.imread(image_path)
        gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #copyMakeBorder har 4 tarafga 8 pixel miqdorda chegaradagi rangni kopiya qilib kengaytiradi.
        #In case raqam chegaraga borib yopishib qolganligi ehtimoli borligi uchun
        gray = cv2.copyMakeBorder(gray, 8,8,8,8,cv2.BORDER_REPLICATE) #BORDER_REPLICATE -kopiya va kengaytirish
        #oq va qoraga o'tkazib olamiz
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        #contorlarnini topib olamiz. Contour bu rasmdagi obyekt chegarasi
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[1] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts,"left-to-right")
        #cnts = sorted(cnts,key=cv2.contourArea,reverse=True) #eng katta aniq bolgan cntlarni sortlab olyapmiz

        #contourni loop qilib har bir raqamni extract qilamiz.
        for c in cnts:
            #contourni bounding rectangle ga jonatib kerakli coordinatalarni olamiz
            (x,y, w, h) = cv2.boundingRect(c)
            roi = gray[y-5:y+h+5, x - x:5 + w + 5]

            #Extracted ROI ekranga chiqadi
            cv2.imshow("ROI", imutils.resize(roi,width=28))
            #Va bizdan key ezishimizni kutadi. Ezilgan klavish esa rasmga label bo'ladi.
            key = cv2.waitKey(0)

            #"`" - bu mobodo biror bir character harf ezilsa ignore qiladi. Chunki bizga faqat nomer label boladi.
            if key == ord("`"):
                print("INFO character rad etildi...")
                continue
            #chr returns character(a string) from an integer (represents unicode code point of the character).
            key = chr(key).upper()

            #dir_pathga annotation output va key huddi path kabi yoziladi. M: agar 1 ni ezsak data/1 kabi comp path yaratiladi
            #dir_path = os.path.sep.join([args["annotation"], key])
            dir_path = os.path.sep.join('annotation', key)

            #Agar mobodo berilgan papka mavjud bolmasa yangisini yaratish
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            #Endi label bolgan rasm faylni datasetga yozishni boshlaymiz.
            count = counts.get(key,1)
            #avval labellanga dir va keyin raqam bilan fayl nomi yozilyapti.
            #zfill(6) ozi bilan hisoblaganda ozigacha 0 bilan toldirib beradi.
            file_name = os.path.sep.join(dir_path, f"{str(count).zfill(6)}.png")
            cv2.imwrite(file_name, roi)

            counts[key] = count + 1
    #control-c ezilganda loopdan chiqish    except KeyboardInterrupt:
        print("INFO skriptdan chiqish")
        break

    except:
        print("INFO rasmni o'tkazib yuborish...")


