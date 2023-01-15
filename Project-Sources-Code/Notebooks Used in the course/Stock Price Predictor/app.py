# from logging im port debug
from urllib import request
from flask import Flask, render_template 
import utils 

def create_app():
    app = Flask(__name__)  
    return app 
# @app.route('/') 
# def home(): 
#     return render_template('index.html') 

# @app.route('/predict/', methods = ["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         Open = request.form.get("Open")  
#         High = request.form.get("High")
#         Low = request.form.get("Low")
#         Volume = request.form.get("Volume")
#     prediction = utils.preprocess(Open, High, Low, Volume)
#     return render_template('prediction.html', prediction = prediction)


if __name__ == "__main__": 
    app.run(debug=True) 
    