from flask import Flask
from .config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    from .routes import bp as pytorch_bp
    app.register_blueprint(pytorch_bp)
    return app
