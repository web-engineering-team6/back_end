from urllib2 import Request, urlopen, HTTPError

#request = Request('https://private-daeedf-aichangesfaces.apiary-mock.com/face_analysis/imageAnalysis?imageId=000001&analysisType=source')

print("~~~~~~~~~~   test   ~~~~~~~~~~")

values = """{}"""
headers = {
  'Content-Type': 'application/json'
}
request = Request('http://127.0.0.1:5000/v1/test', data=values, headers=headers)
#response_body = urlopen(request).read()
#print response_body

print("~~~~~~~~~~   test Face Analysis   ~~~~~~~~~~")

values = """
{
    "image_url": "http://img.hadalove.jp/wp-content/uploads/2016/10/d864633e8b3b05fb233433cc7ab67701-e1476150511960.jpg",
    "analysisType": "seasoning"
}"""

headers = {
  'Content-Type': 'application/json'
}
#request = Request('http://127.0.0.1:5000/v1/faceAnalysis', data=values, headers=headers)
request = Request('http://127.0.0.1:5000/v1/faceAnalysis', data=values, headers=headers)
#https://blooming-caverns-18971.herokuapp.com/

response_body = urlopen(request).read()
print response_body

quit()

print("~~~~~~~~~~   test Face Analysis Error ~~~~~~")

values = """
{
    "image_url": "http://takablog.net/wp-content/uploads/2014/11/okinawa_kuroki.jpg",
    "analysisType": "seasoning"
}"""

headers = {
  'Content-Type': 'application/json'
}
request = Request('http://127.0.0.1:5000/v1/faceAnalysis', data=values, headers=headers)

try:
    response_body = urlopen(request).read()
    print response_body
except HTTPError, e:
    print(e.code)
    print(e.reason)
        
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
