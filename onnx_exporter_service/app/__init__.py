from flask import Flask
from .config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    from .routes import bp as exporter_bp
    app.register_blueprint(exporter_bp)

    return app