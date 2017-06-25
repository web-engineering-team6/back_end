# -*- coding: utf-8 -*-

import json

from flask import Flask, request, jsonify

app = Flask(__name__)
    
@app.route("/v1/faceAnalysis", methods=['GET'])
def faceAnalysis():
    # request.argsにクエリパラメータが含まれている
    imageId = request.args.get("imageId", "000001")
    analysisType = request.args.get("analysisType", "source")
    
    f = open("static/text.txt", "r")
    txt = f.read()
    f.close()
    
    text_json = """
{
    "text": [
        txt
    ]
}
"""
    return text_json
    
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