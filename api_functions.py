#coding: utf-8

import numpy
import requests
import os
import sys
import cv2

# 画像をダウンロードする
def pull_image(url, timeout = 10):
	#urlを引数にして画像を返します。
	response = requests.get(url, allow_redirects=False, timeout=timeout)
	if response.status_code != 200:#もしwebサーバのレスポンスが成功でなかったら
		e = Exception("HTTP status: " + response.status_code)
		raise e
	content_type = response.headers["content-type"]#'text/html'や'image'
	if 'image' not in content_type:#もしcontent_typeに'image'なければ
		e = Exception("Content-Type: " + content_type)
		raise # coding=utf-8
	return response.content

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

def check_faces():
	#画像とfaceTypeを引き数に顔の組成を調べます。
	#切り出した画像が顔かどうか調べる。
	#顔のカスケード分類器のファイルを同じディレクトリに入れてパス指定
	cascade_path = "haarcascade_frontalface_alt.xml"
	#カスケード分類器の特徴量を取得する
	cascade = cv2.CascadeClassifier(cascade_path)
	img = cv2.imread(filename)
	# 顔を検知
	faces = cascade.detectMultiScale(img, minSize=(48,48))#顔miniサイズ仮に48,48とした
	#このコードでは、一つの写真には一人だけ写っている前提
	if faces == cascade.detectMultiScale(img, minSize=(48,48)):
		for (x,y,w,h) in faces:
	    	# 検知した顔を矩い青の四角形で囲む
	    	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#96*96にする方法分からず。
	    	#顔の部分だけのカット写真を作る。
	    	cut_img = img[y:y+h,x:x+w]
			# 画像表示と保存
			cv2.imshow('img',img)
			cv2.imwrite("face_of_" + filename,cut_img)
		#回転不変性もあると嬉しいです。→郷治このやり方分かりません(7月3日(月))。
		#check_facesでしっかりとした顔画像があるかどうか調べて違ったら最低サイズを更新していく感じで作っていきたいです。
		# →郷治このやり方分かりません(7月3日(月))。
	else:
		return None# return Noneは、返した値が後々使われるときに使われます。

#郷治にて上のcheck_faces()に顔検出・出力まで記載したのですが、cut_facesの関数が別に必要である理由がよくわかりません。
def cut_faces():
	#画像の型に関しては迷っています。おそらくndarrayです。→郷治よく理解できません。
	#顔を切り出す。
	#画像を引き数に顔を検出
	#96x96の画像を出力 →郷治:出力画像の大きさを96*96にする方法分かりませんでした。
	return None

#郷治には顔の分析の仕方とそのコード分かりません。
def analysis_face()
	#顔を分析する。
	#返り値はjsonで
	return None

#以下は郷治にてanalysis_typeのスペルミスのみ直しました。
def face_analysis_main(image_url, analysis_type):
	face_image = pull_image(url)
	cut_image = cut_faces(face_image)
	face_component = analysis_faces(cut_image, analysis_type)
	return face_component
