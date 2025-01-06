import unittest
from utils.preprocessor import DataPreprocessor
from transformers import AutoTokenizer
from config import Config

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        config = Config()
        self.tokenizer = AutoTokenizer.from_pretrained(config.DEFAULT_MODEL_NAME)
        self.preprocessor = DataPreprocessor(self.tokenizer)
    
    def test_data_preparation(self):
        # Add your test cases here
        pass