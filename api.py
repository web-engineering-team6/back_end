# -*- coding: utf-8 -*-

import json

from flask import Flask, request, jsonify

app = Flask(__name__)
    
@app.route("/v1/faceAnalysis", methods=['POST'])
def faceAnalysis():
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400
    
    # request.argsにクエリパラメータが含まれている
    image_url = json.loads(request.data)["image_url"]
    analysis_type =  json.loads(request.data)["analysisType"]
        
    f = open(image_url, "r")
    txt = f.read()
    f.close()
    
    num = int(txt[10:13])
    
    return  txt[:9] 
    
@app.route('/v1/changeFace', methods=['POST'])
def changeFace():
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    eachPara = json.loads(request.data)["eachParameterValue"]
    
    sumPara = eachPara[0] + eachPara[1]

    return str(sumPara)


if __name__ == '__main__':
    app.run(debug=True)