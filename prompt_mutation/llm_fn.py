
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import os


@dataclass
class LLMConfig:
    backend: str = "mock"
    model: str = "mock-model"
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0
    seed: Optional[int] = 0
    system_prompt: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    device: str = "cpu"
    do_sample: bool = False


class BaseLLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        lower = prompt.lower()
        if "rewrite the following user query" in lower or "rephrase" in lower:
            src = prompt.split("Query:", 1)[-1].strip()
            return f"Please answer this question carefully: {src}"
        if "paraphrase the following passage" in lower:
            src = prompt.split("Passage:", 1)[-1].strip()
            return f"In other words, {src}"
        if "scientific problem description" in lower or "problem:" in lower:
            src = prompt.split("Problem:", 1)[-1].strip()
            return f"Consider the following problem: {src}"
        return f"[MOCK OUTPUT] {prompt[:200]}"


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        from openai import OpenAI
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key in environment variable: {config.api_key_env}")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        messages: List[Dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            seed=self.config.seed,
        )
        return resp.choices[0].message.content.strip()


class HFLocalClient(BaseLLMClient):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModelForCausalLM.from_pretrained(config.model)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if config.device == "cpu" else 0,
        )

    def generate(self, prompt: str) -> str:
        generated = self.pipe(
            prompt,
            max_new_tokens=self.config.max_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.do_sample else None,
            top_p=self.config.top_p if self.config.do_sample else None,
            return_full_text=False,
        )
        return generated[0]["generated_text"].strip()


def make_client(config: LLMConfig) -> BaseLLMClient:
    if config.backend == "mock":
        return MockLLMClient(config)
    if config.backend == "openai":
        return OpenAIClient(config)
    if config.backend == "hf_local":
        return HFLocalClient(config)
    raise ValueError(f"Unsupported backend: {config.backend}")


def make_llm_fn(config: LLMConfig):
    client = make_client(config)
    return client.generate
