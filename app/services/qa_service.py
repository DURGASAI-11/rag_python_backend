import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QAService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def cosine_similarity(self, a, b):
        return np.dot(a, b)

    def generate_answer(self, question, context):
        prompt = f"""
        Answer the question using only the provided context.
        If answer not found say:
        "Answer not found in the document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
