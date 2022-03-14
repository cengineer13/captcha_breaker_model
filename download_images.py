
import argparse
import requests
import time
import os

#Argument parserni qurib olisgh
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
ap.add_argument("-n", "--num-images", type=int,
	default=500, help="# of images to download")
args = vars(ap.parse_args())

#Captcha mavjud link. Keyinchalik mavjud bo'lmaslik ehtimoli ham bor.
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

# Kiritilgan rasm bo'ylab loop boladi
for i in range(0, args["num_images"]):
	try:
		# try to grab a new captcha image
		r = requests.get(url, timeout=60)

		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(5))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()

		# update the counter
		print("INFO: Yuklandi {}".format(p))
		total += 1


	except:
		print("INFO... Yuklashda xato")

	#Serverga bosim qilmaslik uchun orada pauza belgilash
	time.sleep(0.1)