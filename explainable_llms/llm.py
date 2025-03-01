from typing import Optional, Any, TypeAlias
from transformers import PreTrainedModel, PreTrainedTokenizer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import warnings
import random

LLM_CONFIGS: dict[str, dict[str, Any]] = {
    "llama-3-": {
        "chat_template": "llama-3",
        "eot_strings": ["<|eot_id|>", "<|end_of_text|>"],
    },
    "Meta-Llama-3.1-": {
        "chat_template": "llama-3.1",
        "eot_strings": ["<|eot_id|>", "<|end_of_text|>"],
    },
    "Llama-3.2-": {
        "chat_template": "llama-3.1",
        "eot_strings": ["<|eot_id|>", "<|end_of_text|>"],
    },
    "Mistral-Nemo-Instruct-": {
        "chat_template": "mistral",
        "eot_strings": ["</s>"],
    },
    "mistral-7b-instruct-v0.3-": {
        "chat_template": "mistral",
        "eot_strings": ["</s>"],
    },
    "gemma-2-": {
        "chat_template": "gemma2_chatml",
        "eot_strings": ["<end_of_turn>", "<|im_end|>"],
    },
    "Qwen2.5-7B-Instruct-": {
        "chat_template": None,
        "eot_strings": ["<|im_end|>"],
    },
    "Qwen2-7B-Instruct-": {
        "chat_template": None,
        "eot_strings": ["<|im_end|>"],
    },
    "zephyr-sft-": {
        "chat_template": "zephyr",
        "eot_strings": ["</s>"],
    },
    "phi-4-": {
        "chat_template": "phi-4",
        "eot_strings": ["<|im_end|>"],
    },
    "dummy-model": {
        "chat_template": "gemma2_chatml",
        "eot_strings": ["<end_of_turn>", "<|im_end|>"],
    },
}


class DummyLLM:
    name_or_path: str = "dummy-model"

    def generate(self, **kwargs: Any) -> Any:
        return kwargs["input_ids"]


LLM: TypeAlias = PreTrainedModel | DummyLLM


def get_llm_config(
    llm: Optional[LLM] = None, llm_name_or_path: Optional[str] = None
) -> dict[str, Any]:
    if (llm is None) == (llm_name_or_path is None):
        raise ValueError("Either llm or llm_name_or_path must be provided")
    llm_name_or_path = llm_name_or_path or llm.name_or_path  # type: ignore
    for name, config in LLM_CONFIGS.items():
        if name.lower() in llm_name_or_path.lower():
            return config
    raise ValueError(f"No config available for the llm (or path) {llm_name_or_path}")


def _get_chat_template(tokenizer: PreTrainedTokenizer, llm: LLM) -> PreTrainedTokenizer:
    llm_config = get_llm_config(llm)
    chat_template = llm_config["chat_template"]
    if chat_template is None:
        return tokenizer
    mapping = {
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt",
    }
    return get_chat_template(tokenizer, chat_template=chat_template, mapping=mapping)


def load_pretrained_llm(
    llm_name: str,
    max_seq_length: int = 2048,
    dtype: Any = None,
    load_in_4bit: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=llm_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = _get_chat_template(tokenizer, llm)
    return llm, tokenizer


def convert_llm_to_peft(llm: PreTrainedModel) -> PreTrainedModel:
    return FastLanguageModel.get_peft_model(
        llm,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=random.randint(1, 1000000),
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )


def load_finetuned_llm(
    llm_name: str, max_seq_length: int, dtype: Any, load_in_4bit: bool
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    llm, tokenizer = FastLanguageModel.from_pretrained(
        model_name=llm_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    warnings.warn("Setting llm to inference mode")
    FastLanguageModel.for_inference(llm)
    tokenizer = _get_chat_template(tokenizer, llm)
    return llm, tokenizer
