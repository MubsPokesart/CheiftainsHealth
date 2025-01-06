from flask import Flask
from config import Config
from views import views

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__,
                static_folder='static',
                template_folder='templates')
    
    app.config.from_object(Config)
    
    # Register blueprints
    app.register_blueprint(views)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=Config.DEBUG)
