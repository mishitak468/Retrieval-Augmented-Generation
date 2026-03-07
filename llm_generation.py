"""
LLM Generation Module - FINAL WORKING VERSION
Uses temperature=0.1 for maximum stability with TinyLlama
"""

from llama_cpp import Llama
import os
import re


class LLMGenerator:
    def __init__(self, model_path, model_type='tinyllama', n_ctx=1024, n_threads=4):
        """Initialize LLM - optimized for stability"""
        self.model_type = model_type

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,  # Smaller context for stability
            n_threads=n_threads,
            verbose=False,
            seed=42  # Deterministic
        )

        print(f"✓ Model loaded - Using VERY low temperature for stability")

    def generate(self, prompt, max_tokens=100, **kwargs):
        """
        Generate with MAXIMUM stability settings
        Very low temperature, very restrictive sampling
        """
        formatted_prompt = self._format_prompt(prompt)

        # ULTRA-RESTRICTIVE SETTINGS for stability
        output = self.llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.1,  # VERY LOW - almost deterministic
            top_p=0.5,        # VERY restrictive
            top_k=10,         # Very limited vocabulary
            repeat_penalty=1.3,  # Strong anti-repetition
            echo=False,
            stop=["\n\n", "Q:", "Facts:", "</s>", "\n", "?", "<|"],
            seed=42
        )

        response = output['choices'][0]['text'].strip()
        response = self._clean_response(response)

        return response

    def _format_prompt(self, user_prompt):
        """Minimal formatting - don't confuse small models"""

        if self.model_type == 'tinyllama':
            # Absolute minimum formatting
            return f"""<|user|>
{user_prompt}
<|assistant|>
"""

        elif 'qwen' in self.model_type:
            return f"""<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

        else:
            return f"{user_prompt}\nAnswer: "

    def _clean_response(self, response):
        """Aggressive cleaning"""

        # Remove any tags
        response = re.sub(r'<\|[^>]+\|>', '', response)
        response = re.sub(r'<[^>]+>', '', response)

        # Remove common garbage
        garbage = [
            "Конкретная информация:",
            "Спасибо за ответ",
            "Ответ:",
            "К сожалению",
            "Do you want to continue",
            "Can you repeat",
            "Sorry, but",
            "Would you like me to",
            "Глаза на тебе"
        ]

        for g in garbage:
            if g in response:
                response = response.split(g)[0].strip()

        # Take only first sentence if multiple
        sentences = response.split('. ')
        if sentences:
            response = sentences[0]
            if not response.endswith('.'):
                response += '.'

        # Clean whitespace
        response = re.sub(r'\s+', ' ', response).strip()

        # If too short or empty, return fallback
        if len(response) < 15:
            response = "Based on the provided context, I cannot generate a clear answer."

        return response

    def create_augmented_prompt(self, query, retrieved_docs):
        """
        ABSOLUTE MINIMUM prompt format
        """
        # Take ONLY the most relevant document
        if not retrieved_docs:
            return f"Q: {query}\nA:"

        # Use only top document, first 100 chars
        top_doc = retrieved_docs[0][2][:100].strip() + "..."

        # Ultra-simple format
        prompt = f"""{top_doc}

Question: {query}
Answer:"""

        return prompt


def test():
    """Quick test"""
    import glob

    models = glob.glob("*.gguf")
    if not models:
        print("No model found")
        return

    model_path = models[0]
    model_type = 'qwen2-1.5b' if 'qwen' in model_path.lower() else 'tinyllama'

    print(f"Testing with: {model_path}")

    gen = LLMGenerator(model_path, model_type)

    test_prompt = """Business associations are membership organizations engaged in promoting business interests.

Question: What is a business association?
Answer:"""

    response = gen.generate(test_prompt, max_tokens=50)
    print(f"\nResponse: {response}")


if __name__ == "__main__":
    test()
