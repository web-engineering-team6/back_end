from urllib2 import Request, urlopen, HTTPError

#request = Request('https://private-daeedf-aichangesfaces.apiary-mock.com/face_analysis/imageAnalysis?imageId=000001&analysisType=source')

print("~~~~~~~~~~   test   ~~~~~~~~~~")

values = """{}"""
headers = {
  'Content-Type': 'application/json'
}
request = Request('https://blooming-caverns-18971.herokuapp.com/v1/test', data=values, headers=headers)
response_body = urlopen(request).read()
print response_body

print("~~~~~~~~~~   test Face Analysis   ~~~~~~~~~~")

values = """
{
    "image_url": "https://scontent-nrt1-1.xx.fbcdn.net/v/t1.0-1/p160x160/10487422_859730147372143_7945100787930646381_n.jpg?oh=4b1550c4ba26914cf6739557443a9d92&oe=59C57D46",
    "analysis_type": "seasoning"
}"""

headers = {
  'Content-Type': 'application/json'
}
#request = Request('http://127.0.0.1:5000/v1/faceAnalysis', data=values, headers=headers)
request = Request('https://blooming-caverns-18971.herokuapp.com/v1/faceAnalysis', data=values, headers=headers)
#https://blooming-caverns-18971.herokuapp.com/

response_body = urlopen(request).read()
print response_body

print("~~~~~~~~~~   test Face Analysis Error ~~~~~~")

values = """
{
    "image_url": "https://qiita-image-store.s3.amazonaws.com/0/53147/87053ada-2d90-6aa5-3531-ed3b54ffad49.png",
    "analysis_type": "seasoning"
}"""

headers = {
  'Content-Type': 'application/json'
}
request = Request('https://blooming-caverns-18971.herokuapp.com/v1/faceAnalysis', data=values, headers=headers)

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
