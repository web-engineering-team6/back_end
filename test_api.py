from urllib2 import Request, urlopen

#request = Request('https://private-daeedf-aichangesfaces.apiary-mock.com/face_analysis/imageAnalysis?imageId=000001&analysisType=source')

print("~~~~~~~~~~   test Face Analysis   ~~~~~~~~~~")

request = Request('http://127.0.0.1:5000/v1/faceAnalysis?imageId=000001&analysisType=source')

response_body = urlopen(request).read()
print response_body

print("~~~~~~~~~~   test Cange Face   ~~~~~~~~~~")

values = """
{
    "eachParameterValue": [
        1.23,
        2.34,
        3.45,
        6.78
      ]
}
"""
headers = {
  'Content-Type': 'application/json'
}
request = Request('http://127.0.0.1:5000/v1/changeFace', data=values, headers=headers)

response_body = urlopen(request).read()
print response_body
