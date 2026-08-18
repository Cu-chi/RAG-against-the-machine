from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import torch


logging.set_verbosity_error()


class LLMGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16
            if torch.cuda.is_available() else torch.float32,
        )

    def generate_answer(self, query: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."
                                          "Answer based on the provided"
                                          "context only."},
            {"role": "user", "content":
                f"""Use the following pieces of retrieved context to answer
the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and strictly based on the context.

Context:
{context}

Question: {query}
Answer:"""}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer([text],
                                return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.4,
            top_p=0.8,
            top_k=20,
            min_p=0.0
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids,
                                               skip_special_tokens=True)[0]
        return response.strip()
