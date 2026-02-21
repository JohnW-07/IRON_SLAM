import os
import threading
from flask import Flask, request, jsonify, render_template
from models import db, Video
from ml.video_analyzer import analyze_video

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    video = Video(filename=file.filename)
    db.session.add(video)
    db.session.commit()

    thread = threading.Thread(target=process_video, args=(video.id, filepath))
    thread.start()

    return jsonify({"video_id": video.id})

@app.route('/status/<int:video_id>')
def check_status(video_id):
    video = Video.query.get(video_id)
    return jsonify({
        "status": video.status,
        "results": video.results
    })

@app.route('/results/<int:video_id>')
def results_page(video_id):
    video = Video.query.get(video_id)
    return render_template("results.html", video=video)

def process_video(video_id, filepath):
    with app.app_context():
        results = analyze_video(filepath)
        video = Video.query.get(video_id)
        video.status = "done"
        video.results = results
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True)