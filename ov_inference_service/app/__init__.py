from flask import Flask
from .config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    from .routes import bp as ov_inference_bp
    app.register_blueprint(ov_inference_bp)

    return app
