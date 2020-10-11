from flask import Flask, request, render_template
import re
import requests
import base64
import boto3  # for AWS
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from tensorflow.keras.models import load_model  # load the FBP model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imresize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Functions
# TODO ReX needed to be changed
# def code2text(embeded_code):
#     res = re.findall(r"<p.*?>(.*?)</p>", embeded_code)
#     if res:
#         if re.findall(r"<a.*?>", res[0]):
#             res = re.findall(r"(.*?)<a.*?>", res[0])
#         res = res[0]
#     else:
#         res = ''
#     return res


def twiiter_rumor(tweet_text):
    url = "https://o2k4cwfco4.execute-api.us-west-2.amazonaws.com/prod/twiiter"
    data = {
        "twitter": tweet_text,
    }
    resp = requests.get(url, params=data)

    if resp.status_code == 200:
        res = resp.json()
        return res


def twitter_keyPhrase(tweet_text):
    url = "https://ly16hu4xd4.execute-api.us-west-2.amazonaws.com/Prod/twitter"
    data = {
        "twitter": tweet_text,
    }
    resp = requests.get(url, params=data)
    if resp.status_code == 200:
        res = resp.json()
        return res


def twiiter_Sentiment(tweet_text):
    url = "https://udl04bvxgf.execute-api.us-west-2.amazonaws.com/Prod/twitter"
    data = {
        "twitter": tweet_text,
    }
    resp = requests.get(url, params=data)
    if resp.status_code == 200:
        res = resp.json()
        return res


# Set the basic parameters for images
img_height, img_width, channels = 350, 350, 3
########################################################################################################################
# Load the model from S3 bucket
BUCKET_NAME = 'cv2bucketlololo28y43875t47'
MODEL_FILE_NAME = 'CV.h5'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME

# Build connection with S3 bucket
conn = S3Connection()
bucket = conn.get_bucket(BUCKET_NAME, validate=False)
key_obj = Key(bucket)
key_obj.key = MODEL_FILE_NAME

contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
FBP_model = load_model(MODEL_LOCAL_PATH)
########################################################################################################################

application = Flask(__name__)


# A welcome page for using our ML service.
@application.route('/')
@application.route('/hello')
def hello():
    return render_template('hello.html')


# For NLP service
@application.route('/nlp', methods=["POST", "GET"])
def nlp():
    default_response = 'Tweet Text Needed'
    response0 = default_response
    response1 = default_response
    response2 = []
    percent1 = 0
    percent2 = 0
    percent3 = 0

    # Here are some NLP APIs for the analysis.
    if request.method == 'POST':
        if request.form['tweet_code']:

            # embedded_code = request.form['tweet_code']
            # tweet_text = code2text(embedded_code)
            tweet_text = request.form['tweet_code'].replace('#', '')
            response0 = tweet_text
            if tweet_text:
                response1 = twiiter_rumor(tweet_text)
                response2 = list(twitter_keyPhrase(tweet_text))
                temp = twiiter_Sentiment(tweet_text)[0]['confidenceScores']
                temp = list(temp.values())
                percent1 = int(temp[0]*100)
                percent2 = int(temp[1]*100)
                percent3 = int(temp[2]*100)

    return render_template('nlp.html', response0=response0, response1=response1,
                           response2=response2,
                           percent1=percent1, percent2=percent2, percent3=percent3)


# For FBP service
@application.route('/cv', methods=['POST', 'GET'])
def fbp():
    default_score = 0
    score = default_score
    if 'file' in request.files and request.method == 'POST':  # The picture is uploaded successfully
        # Get the uploaded image:
        # 1. Transform filestorage file to base64 file
        b64image = base64.b64encode(request.files['file'].read())
        # 2. Transform base64 file to image file
        with open("image.png", "wb") as f:
            f.write(base64.b64decode(b64image))
            image_for_model = imresize(load_img("image.png"), size=(img_height, img_width))
            image = img_to_array(image_for_model).reshape(img_height, img_width, channels)
            image /= 255.0
            image = image.reshape((1,)+image.shape)
            # Using the above image_path to make prediction
            score = round(FBP_model.predict(image)[0][0], 3)
            f.close()
        # Delete the file after usage!
        os.remove("image.png")

    return render_template('fbp.html', score=score)


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080, debug=True, threaded=False)