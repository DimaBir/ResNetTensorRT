from flask import Flask
from ..config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register routes
    from .routes import bp as core_logic_bp
    app.register_blueprint(core_logic_bp)

    return app