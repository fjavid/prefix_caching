from __future__ import annotations

from typing import Optional

from .backend_base import BackendBase, GenerationResult


class SGLangBackend(BackendBase):
    """
    Placeholder SGLang backend.
    The interface is ready, but the implementation is deferred.
    """

    def start(self) -> None:
        raise NotImplementedError('SGLang backend is scaffolded but not implemented yet.')

    def stop(self) -> None:
        self._started = False

    def generate(self, prompt: str, request_id: Optional[str] = None) -> GenerationResult:
        raise NotImplementedError('SGLang backend is scaffolded but not implemented yet.')

    def backend_name(self) -> str:
        return 'sglang'
