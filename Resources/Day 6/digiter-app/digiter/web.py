from flask import Flask, request, jsonify
from digiter.helpers.digiter_helper import get_digit

app = Flask(__name__)

@app.route('/health', methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route('/digit', methods=["POST"])
def get_digit_route():
    image_base64 = request.json.get("image")
    digit_dict = get_digit(image_base64)
    return jsonify(digit_dict)

