from mok.__main__ import build_default_backends
from mok.models.backends import HTTPBackend, MockBackend
from mok.models.backends_llama import LlamaCppBackend
from mok.models.backends_ollama import OllamaBackend


def test_cli_registers_real_local_backends() -> None:
    backends = build_default_backends()

    assert isinstance(backends["local"], MockBackend)
    assert isinstance(backends["mock"], MockBackend)
    assert isinstance(backends["vllm"], MockBackend)
    assert isinstance(backends["http"], HTTPBackend)
    assert isinstance(backends["ollama"], OllamaBackend)
    assert isinstance(backends["llama_cpp"], LlamaCppBackend)
