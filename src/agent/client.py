import os
import instructor
from openai import OpenAI
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class InstructorClient:
    """
    Wrapper for the instructor client pointing to the H200 vLLM server.
    Uses 'instructor.from_openai' for robust JSON schema enforcement.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "liuclaw-local-key"):
        # Configure instructor to wrap OpenAI for vLLM JSON mode
        self.client = instructor.from_openai(
            OpenAI(
                base_url=base_url,
                api_key=api_key,
            ),
            mode=instructor.Mode.JSON
        )

    def get_structured_response(self, response_model: Type[T], system_prompt: str, user_prompt: str, model_name: str = "Qwen/Qwen2.5-32B-Instruct") -> T:
        """Fetch a guaranteed parsed Pydantic object from vLLM."""
        return self.client.chat.completions.create(
            model=model_name,
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1 # Keep it deterministic for mathematical reasoning
        )
