#coding: utf-8

import numpy as np
import requests
import os
import sys
import cv2
from deep_learnning.conv_net import *

# 画像をダウンロードする
def pull_image(url, timeout = 10):
	#urlを引数にして画像を返します。
	response = requests.get(url, allow_redirects=False, timeout=timeout)
	if response.status_code != 200:
		e = Exception("HTTP status: " + str(response.status_code))
		raise e
	content_type = response.headers["content-type"]
	if 'image' not in content_type:
		e = Exception("Content-Type: " + content_type)
		raise # coding=utf-8
	buf = np.fromstring(response.content, dtype=np.uint8)
	face = cv2.imdecode(buf, cv2.IMREAD_COLOR)
	return face

# 画像のファイル名を決める
def make_filename(base_dir, number, url):
    ext = os.path.splitext(url)[1] # 拡張子を取得
    filename = number + ext        # 番号に拡張子をつけてファイル名にする
    fullpath = os.path.join(base_dir, filename)
    return fullpath

# 画像を保存する
def save_image(filename, image):
    with open(filename, "wb") as fout:
        fout.write(image)

def check_faces(cut_face):
	print("a")
	return True# return Noneは、返した値が後々使われるときに使われます。

def cut_faces(face):
	#画像の型に関しては迷っています。おそらくndarrayです。→郷治よく理解できません。
	#顔を切り出す。
	#画像を引き数に顔を検出
	#96x96の画像を出力 →郷治:出力画像の大きさを96*96にする方法分かりませんでした。
	cascade_path = "haarcascade_frontalface_alt.xml"
	cascade = cv2.CascadeClassifier(cascade_path)
	center = tuple(np.array(face.shape[0:2])/2)
	rIntr = 15
	rs = 0
	re = 360
	for r in range(rs, re+1, rIntr):
		rotMat = cv2.getRotationMatrix2D(center, r, 1.0)    
		roll_face = cv2.warpAffine(face, rotMat, face.shape[0:2], flags=cv2.INTER_LINEAR)
		faces = cascade.detectMultiScale(roll_face, minSize=(45,45))
		#cv2.imshow('img',roll_face)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		print(faces)
		for (x,y,w,h) in faces:
			cut_face = roll_face[y:y+h,x:x+w]
			cut_face = cv2.resize(cut_face,(96,96))
			if check_faces(cut_face):
				return cut_face
	

#郷治には顔の分析の仕方とそのコード分かりません。
def analysis_face(face_image, analysis_type="seasoning"):
	network = load_network("network_10")
	face_data = network.predict(face_image)
	print(face_data)
	#返り値はjsonで
	return None

#以下は郷治にてanalysis_typeのスペルミスのみ直しました。
def face_analysis_main(image_url, analysis_type):
	face_image = pull_image(url)
	cut_image = cut_faces(face_image)
	face_component = analysis_faces(cut_image, analysis_type)
	return face_component

url = "https://img.cinematoday.jp/a/N0087026/_size_1200x/_v_1477451400/main.jpg"
face = pull_image(url, timeout = 10)
cut_face = cut_faces(face)
cv2.imshow('img',cut_face)
cv2.waitKey(0)
cv2.destroyAllWindows()

analysis_face(cut_face, analysis_type="seasoning")
