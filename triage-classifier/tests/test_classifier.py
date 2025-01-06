import unittest
from models.classifier import TriageClassifier

class TestTriageClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = TriageClassifier()
    
    def test_prediction(self):
        text = "I'm experiencing severe chest pain"
        result = self.classifier.predict(text)
        
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn(result["label"], ["urgent", "non-urgent"])
        self.assertTrue(0 <= result["confidence"] <= 1)
