from flask import Flask
from views import views
from config import Config


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
