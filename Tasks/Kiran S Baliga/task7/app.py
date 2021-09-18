import numpy as np
from flask import Flask, request, jsonify, render_template
import picklefrom keras.models import load_model

# Create flask app
flask_app = Flask(__name__)
model = load_model('model.h5')

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():

    img=request.form['img']
    img=np.asarray(img)
    prediction = model.predict(img)
    return render_template("index.html", prediction_text = "The class is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)