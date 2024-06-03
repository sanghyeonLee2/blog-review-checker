from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
  
    from app.routes.blog import blog_bp
    from app.routes.process import process_bp


    app.register_blueprint(blog_bp)
    app.register_blueprint(process_bp)

    return app
