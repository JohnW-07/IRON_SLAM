from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    status = db.Column(db.String(50), default="processing")
    results = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)