from flask import Flask, request, render_template, url_for
import re
import requests


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


application = Flask(__name__)
# Some basic setting for uploading image files



# A welcome page for using our ML service.
@application.route('/')
@application.route('/hello')
def hello():
    return render_template('hello.html')


# For NLP service
@application.route('/nlp', methods=["POST", "GET"])
def nlp():
    default_response = 'You should input the Embedded CODE from the exact tweet!'
    response0 = default_response
    response1 = default_response
    response2 = default_response
    response3 = default_response

    # here is some NLP APIs for the analysis.
    if request.method == 'POST':
        if request.form['tweet_code']:
            embedded_code = request.form['tweet_code']
            tweet_text = code2text(embedded_code)
            response0 = tweet_text
            if tweet_text:
                # Backend API needed!
                response1 = twiiter_rumor(tweet_text)
                response2 = twitter_keyPhrase(tweet_text)
                response3 = twiiter_Sentiment(tweet_text)[0]['confidenceScores']

    return render_template('nlp.html', response0=response0, response1=response1, response2=response2, response3=response3)


# For FBP service
@application.route('/fbp', methods=['POST', 'GET'])
def fbp():
    default_score = 0
    if request.form == 'POST':
        if request.files['file']:
            score = 5
            print('change')
        return render_template('fbp.html', score=score)
    else:
        return render_template('fbp.html', score=default_score)


if __name__ == '__main__':
    application.run(debug=True)