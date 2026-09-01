"""Module with the class for the LLM generation."""
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar
import torch
from typing import Any


class LLMGenerator:
    """Class used for the generation."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the LLMGenerator class.

        Args:
            model_name (str, optional): Model name to load using HF.
            Defaults to "Qwen/Qwen3-0.6B".
        """
        print("Loading model...")
        disable_progress_bar()
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16
            if torch.cuda.is_available() else torch.float32,
        )

    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer for the query using the model.

        Args:
            query (str): the query to answer.
            context (str): context for the model.

        Returns:
            str: the answer using the context
        """
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."
                                          " Answer based on the provided "
                                          "context only."},
            {"role": "user", "content":
                f"""Use the following pieces of retrieved context to answer
the question.
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
        response: str = self.tokenizer.decode(generated_ids,
                                              skip_special_tokens=True)[0]
        return response.strip()
