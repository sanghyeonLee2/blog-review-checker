from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
  
    from app.routes.fetch_from_outer import fetch_from_outer_bp
    from app.routes.fetch_from_inner import fetch_from_inner_bp


    app.register_blueprint(fetch_from_outer_bp)
    app.register_blueprint(fetch_from_inner_bp)

    return app
