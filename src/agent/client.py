import os
import instructor
from openai import OpenAI
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class InstructorClient:
    """
    Wrapper for the instructor client pointing to the H200 vLLM server.
    """
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "empty"):
        # H200 vLLM server defaults to localhost:8000 if not specified
        self.client = instructor.patch(
            OpenAI(
                base_url=base_url,
                api_key=api_key,
            ),
            mode=instructor.Mode.JSON
        )

    def get_structured_response(self, response_model: Type[T], system_prompt: str, user_prompt: str) -> T:
        """Fetch a structured response from the LLM."""
        return self.client.chat.completions.create(
            model="h200-model", # Placeholder for the actual model name on H200
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0 # Greedy for deterministic mutations
        )
