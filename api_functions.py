#coding: utf-8

import numpy

def pull_image(url):
	#urlを引数にして画像を返します。
	return None

def check_faces():
	#画像とfaceTypeを引き数に顔の組成を調べます。
	#切り出した画像が顔かどうか調べる。
	return None

def cut_faces():
	#画像の型に関しては迷っています。おそらくndarrayです。
	#顔を切り出す。
	#画像を引き数に顔を検出
	#96x96の画像を出力
	#回転不変性もあると嬉しいです。
	#check_facesでしっかりとした顔画像があるかどうか調べて違ったら最低サイズを更新していく感じで作っていきたいです。
	return None
	
def analysis_face()
	#顔を分析する。
	#返り値はjsonで
	return None
	
def face_analysis_main(image_url, analusis_type):
	face_image = pull_image(url)
	cut_image = cut_faces(face_image)
	face_component = analysis_faces(cut_image, analysis_type)
	return face_component