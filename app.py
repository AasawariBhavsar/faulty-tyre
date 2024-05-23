from flask import Flask, request, render_template
from model import load_trained_model, prepare_image
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_trained_model('../models/mobilenet.h5')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            image = prepare_image(file, target_size=(224, 224))
            prediction = model.predict(image)
            result = "Good" if prediction[0] > 0.5 else "Defective"
            return render_template("result.html", result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
