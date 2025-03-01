import pytest
from transformers import PreTrainedTokenizer

from explainable_llms.llm import LLM, load_pretrained_llm


@pytest.fixture(scope="session")  # type: ignore
def llm_data() -> tuple[LLM, PreTrainedTokenizer]:
    llm, tokenizer = load_pretrained_llm("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
    return llm, tokenizer


@pytest.fixture(scope="session")  # type: ignore
def llm(llm_data: tuple[LLM, PreTrainedTokenizer]) -> LLM:
    llm, _ = llm_data
    return llm


@pytest.fixture(scope="session")  # type: ignore
def tokenizer(llm_data: tuple[LLM, PreTrainedTokenizer]) -> PreTrainedTokenizer:
    _, tokenizer = llm_data
    return tokenizer
