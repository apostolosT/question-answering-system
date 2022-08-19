from flask import Flask
from api import api


def create_app(settings_overrides=None):
    app = Flask(__name__)
    configure_settings(app, settings_overrides)
    configure_blueprints(app)
    return app


def configure_settings(app, settings_override):
    app.config.update({
        'DEBUG': True,
        'TESTING': False,
        'SWAGGER': {
            'title': 'QA System App',
            'uiversion': 3,
            'openapi': '3.0.2',
        }
    })
    if settings_override:
        app.config.update(settings_override)


def configure_blueprints(app):
    app.register_blueprint(api)