from flask import Flask, render_template, request, jsonify
import os
import time

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    video = request.files["video"]
    
    if video:
        path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
        video.save(path)

        # Fake ML delay
        time.sleep(2)

        return jsonify({
            "ply_url": "/static/output/snow_pointcloud.glb"
        })

    return jsonify({"error": "No file"}), 400


if __name__ == "__main__":
    app.run(debug=True)