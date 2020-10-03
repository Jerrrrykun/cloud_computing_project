from flask import Flask, request, render_template, send_from_directory, url_for
import re
import requests
# from flask_uploads import UploadSet, configure_uploads, IMAGES
import os


# UPLOAD_FOLDER = ''

# Functions
def code2text(embeded_code):
    res = re.findall(r"<p.*?>(.*?)</p >", embeded_code)
    if res:
        if re.findall(r"<a.*?>", res[0]):
            res = re.findall(r"(.*?)<a.*?>", res[0])
        res = res[0]
    else:
        res = ''
    return res


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


app = Flask(__name__)
# Some basic setting for uploading image files
# photos = UploadSet('photos', IMAGES)

# app.config['UPLOADED_FOLDER'] = UPLOAD_FOLDER
# configure_uploads(app, photos)


# A welcome page for using our ML service.
@app.route('/')
@app.route('/hello')
def hello():
    return render_template('hello.html')


# For NLP service
@app.route('/nlp', methods=["POST", "GET"])
def nlp():
    default_response = 'You should input the Embedded CODE from the exact tweet!'
    response1 = default_response
    response2 = default_response
    response3 = default_response

    # here is some NLP APIs for the analysis.
    if request.method == 'POST':
        if request.form['tweet_code']:
            embedded_code = request.form['tweet_code']
            tweet_text = code2text(embedded_code)
            if tweet_text:
                # Backend API needed!
                response1 = twiiter_rumor(tweet_text)
                response2 = twitter_keyPhrase(tweet_text)
                response3 = twiiter_Sentiment(tweet_text)[0]['confidenceScores']

        return render_template('nlp.html', response1=response1, response2=response2, response3=response3)

    else:
        return render_template('nlp.html', response1=default_response, response2=default_response, response3=default_response)


# For FBP service
@app.route('/fbp')
def fbp():
    score = 5
    # default_pic_path = './sample_img.png'
    # filename = default_pic_path
    # if request.method == 'POST' and 'photo' in request.files:
    #     filename = 'http://127.0.0.1:5000/'+filename

    return render_template('fbp.html', score=score)


# @app.route('/fbp/<filename>')
# def send_image(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)