from flask import Blueprint, render_template, request, jsonify
from models.classifier import TriageClassifier
from config import Config

views = Blueprint('views', __name__)

try:
    classifier = TriageClassifier()
except Exception as e:
    print(f"Error initializing classifier: {str(e)}")
    classifier = None

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        name = request.form['name']
        message = request.form['inquiry']
        
        if classifier is None:
            return jsonify({"error": "Model not available"}), 500
        
        try:
            result = classifier.predict(message)
            triage_status = result['label']
            
            return render_template('results.html', 
                                 message=message,
                                 status=triage_status,
                                 name=name)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return render_template('index.html')