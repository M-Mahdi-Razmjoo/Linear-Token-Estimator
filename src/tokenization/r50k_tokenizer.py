import tiktoken

class Tokenizer():
    def __init__(self):
        self.encoder = tiktoken.get_encoding("r50k_base")
        
    def token_count(self, text: str) -> int:
        """Count the number of tokens in a given text using the r50k tokenizer."""
        return len(self.encoder.encode(text, allowed_special="all"))