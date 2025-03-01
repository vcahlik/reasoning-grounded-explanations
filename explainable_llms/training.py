import torch
import json
import logging
import random
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainerCallback,
    # EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerState
from datasets import Dataset, DatasetDict
from unsloth import is_bfloat16_supported
from enum import Enum
from typing import Any, Mapping, Sequence, Optional

from explainable_llms.llm import LLM
from explainable_llms.problem import Problem
from explainable_llms.utils import DatasetType


class ResponseMode(Enum):
    DECISION = "DECISION"
    EXPLANATION = "EXPLANATION"
    REASONING = "REASONING"


def formatting_prompts_func(
    examples: Mapping[str, Any], tokenizer: PreTrainedTokenizer
) -> dict[str, Any]:
    formatted_columns = {}
    for key in examples.keys():
        if not key.endswith("conversation"):
            continue
        conversations = examples[key]
        texts = [
            tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            for conversation in conversations
        ]
        formatted_columns[key] = texts
    return {"text": [x for xs in zip(*formatted_columns.values()) for x in xs]}


def example_dataset_mapping_fn(
    x: Mapping[str, Any], dataset_type: DatasetType, problem: Problem
) -> Mapping[str, Any]:
    return build_finetuning_dataset(x, dataset_type=dataset_type, problem=problem)


def show_gpu_stats() -> None:
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


def show_trainer_stats(trainer_stats: TrainerState) -> None:
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


def train_llm(
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict,
    max_seq_length: int = 5000,
    max_steps: int = -1,
    per_device_train_batch_size: int = 2,
    learning_rate: float = 0.0001,
    # early_stopping_patience: int = 5,
    eval_steps: int = 100,
    log_path: Optional[str] = None,
) -> None:
    class LoggingCallback(TrainerCallback):  # type: ignore
        def __init__(self, log_path: str):
            super().__init__()
            self.log_path = log_path

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore
            if logs is not None:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    # callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
    callbacks = []
    if log_path is not None:
        callbacks.append(LoggingCallback(log_path=log_path))

    trainer = SFTTrainer(
        model=llm,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            max_steps=max_steps,
            num_train_epochs=1,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            output_dir="outputs",
            logging_steps=eval_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # lower eval loss is better
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=random.randint(0, 1000000),
            report_to="none",
        ),
        callbacks=callbacks,
    )
    show_gpu_stats()
    trainer_stats = trainer.train()
    show_trainer_stats(trainer_stats)


def _get_inference_outputs(
    conversations: Sequence[Any],
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    llm_config: Mapping[str, Any],
    max_new_tokens: int,
) -> list[str]:
    instructs = []
    for conversation in conversations:
        conversation = conversation[:-1]
        txt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,  # Must add for generation
        )
        instructs.append(txt)

    # from transformers import pipeline
    # generator = pipeline(model=llm, tokenizer=tokenizer)
    # outputs = generator(instructs, do_sample=False, return_full_text=False, max_new_tokens=max_new_tokens)

    inputs = tokenizer(instructs, return_tensors="pt", padding=True).to("cuda")
    raw_outputs = llm.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False
    )

    outputs = []
    for i, raw_output in enumerate(raw_outputs):
        response = tokenizer.decode(raw_output[len(inputs[i]) :])
        for eot_string in llm_config["eot_strings"]:
            response = response.replace(eot_string, "")
        outputs.append(response)
    return outputs


def add_batch_inference(
    batch: Dataset,
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    llm_config: Mapping[str, Any],
    max_new_tokens: int = 256,
    disturb_reasoning: bool = False,
) -> Dataset:
    reasoning_outputs = None
    reasoning_conversations_key = f"{ResponseMode.REASONING.value.lower()}_conversation"
    if reasoning_conversations_key in batch:
        reasoning_conversations = batch[reasoning_conversations_key]
        reasoning_outputs = _get_inference_outputs(
            reasoning_conversations, llm, tokenizer, llm_config, max_new_tokens
        )
        if disturb_reasoning:
            batch[f"{ResponseMode.REASONING.value.lower()}_output_original"] = reasoning_outputs
            for i in range(len(reasoning_outputs)):
                reasoning_outputs[i] = ','.join(str(1 - int(bit)) if random.random() < 0.1 else bit for bit in reasoning_outputs[i].split(','))
        batch[f"{ResponseMode.REASONING.value.lower()}_output"] = reasoning_outputs
    response_modes = []
    if "decision_conversation" in batch:
        response_modes.append(ResponseMode.DECISION)
    if "explanation_conversation" in batch:
        response_modes.append(ResponseMode.EXPLANATION)
    for response_mode in response_modes:
        response_mode_name = response_mode.value.lower()
        conversations = batch[f"{response_mode_name}_conversation"]
        if reasoning_outputs is not None:
            inferred_reasoning_conversations = []
            for conversation, reasoning_output in zip(conversations, reasoning_outputs):
                conversation[-3]["value"] = reasoning_output
                inferred_reasoning_conversations.append(conversation)
            conversations = inferred_reasoning_conversations
        outputs = _get_inference_outputs(
            conversations, llm, tokenizer, llm_config, max_new_tokens
        )
        batch[f"{response_mode_name}_output"] = outputs
    return batch


def get_pretrained_few_shot_prompt(
    example: Mapping[str, Any],
    problem: Problem,
    response_mode: ResponseMode,
    example_xs: Optional[Sequence[Sequence[float]]] = None,
    n_example_xs: Optional[int] = None,
    noise_ratio: float = 0.0,
    reveal_classifier_type: bool = True,
) -> str:
    if (example_xs is not None) and (n_example_xs is not None):
        raise ValueError("example_xs and n_example_xs can't be both specified")
    if example_xs is None and n_example_xs is not None:
        example_xs = problem.generate_xs(n_example_xs)
    if response_mode == ResponseMode.DECISION:
        return problem.get_few_shot_decision_prompt(
            example=example,
            example_xs=example_xs,
            noise_ratio=noise_ratio,
            reveal_classifier_type=reveal_classifier_type,
        )
    elif response_mode == ResponseMode.EXPLANATION:
        return problem.get_few_shot_explanation_prompt(
            example=example,
            example_xs=example_xs,
            noise_ratio=noise_ratio,
        )
    else:
        raise ValueError(f"Unknown response mode: {response_mode}")


def get_pretrained_structural_prompt(
    example: Mapping[str, Any],
    problem: Problem,
    response_mode: ResponseMode,
    example_xs: Optional[Sequence[Sequence[float]]] = None,
    n_example_xs: Optional[int] = None,
    noise_ratio: float = 0.0,
) -> str:
    if (example_xs is None) == (n_example_xs is None):
        raise ValueError("Either example_xs or n_example_xs must be provided")
    elif example_xs is None:
        example_xs = problem.generate_xs(n_example_xs)  # type: ignore
    if response_mode == ResponseMode.DECISION:
        return problem.get_structural_decision_prompt(
            example=example,
            example_xs=example_xs,
            noise_ratio=noise_ratio,
        )
    elif response_mode == ResponseMode.EXPLANATION:
        return problem.get_structural_explanation_prompt(
            example=example,
            example_xs=example_xs,
            noise_ratio=noise_ratio,
        )
    else:
        raise ValueError(f"Unknown response mode: {response_mode}")


def build_finetuning_dataset(
    example: Mapping[str, Any],
    dataset_type: DatasetType,
    problem: Problem,
    response_modes: Optional[Sequence[ResponseMode]] = None,
) -> Mapping[str, Any]:
    def process_example(
        example: dict[str, Any],
        dataset_type: DatasetType,
        problem: Problem,
        response_mode: ResponseMode,
        conversation: list[Any],
    ) -> dict[str, Any]:
        if response_mode == ResponseMode.DECISION:
            decision = problem.get_decision(example)
            conversation.extend(
                [
                    {"from": "human", "value": "ANSWER"},
                    {"from": "gpt", "value": decision},
                ]
            )
            example["decision_target"] = decision
            return example
        if response_mode == ResponseMode.EXPLANATION:
            explanation = problem.get_explanation(example)
            conversation.extend(
                [
                    {"from": "human", "value": "EXPLAIN"},
                    {"from": "gpt", "value": explanation},
                ]
            )
            example["explanation_target"] = explanation
            if "decision_target" not in example:
                decision = problem.get_decision(example)
                example["decision_target"] = decision
            return example
        if response_mode == ResponseMode.REASONING:
            example["reasoning_target"] = reasoning
            return example
        raise NotImplementedError

    problem.notify_build_next_finetuning_dataset_instance(
        example_dataset_mapping_fn=example_dataset_mapping_fn,
        dataset_type=dataset_type,
    )
    example = dict(example)
    response_modes = response_modes or list(ResponseMode)
    base_conversation = [
        {"from": "human", "value": problem.get_human_input(example)},
    ]
    if ResponseMode.REASONING in response_modes:
        reasoning = problem.get_reasoning(example)
        base_conversation.append({"from": "gpt", "value": reasoning})
    for response_mode in response_modes:
        conversation = base_conversation.copy()
        example = process_example(
            example, dataset_type, problem, response_mode, conversation
        )
        example[f"{response_mode.value.lower()}_conversation"] = conversation
    # del example["decision"]
    return example


def build_prompting_dataset(
    example: Mapping[str, Any],
    problem: Problem,
    use_problem_structure: bool,
    response_modes: Optional[Sequence[ResponseMode]] = None,
    example_xs: Optional[Sequence[Sequence[float]]] = None,
    n_example_xs: Optional[int] = None,
    noise_ratio: float = 0.0,
    reveal_classifier_type: bool = True,
) -> Mapping[str, Any]:
    example = dict(example)
    response_modes = response_modes or list(ResponseMode)
    modified_example = example.copy()
    for response_mode in response_modes:
        if not use_problem_structure:
            prompt = get_pretrained_few_shot_prompt(
                example,
                problem,
                response_mode=response_mode,
                example_xs=example_xs,
                n_example_xs=n_example_xs,
                noise_ratio=noise_ratio,
                reveal_classifier_type=reveal_classifier_type,
            )
        else:
            prompt = get_pretrained_structural_prompt(
                example,
                problem,
                response_mode=response_mode,
                example_xs=example_xs,
                n_example_xs=n_example_xs,
                noise_ratio=noise_ratio,
            )
        if response_mode == ResponseMode.DECISION:
            target_output = problem.get_decision(example)
        elif response_mode == ResponseMode.EXPLANATION:
            target_output = problem.get_explanation(example)
        else:
            raise ValueError(f"Unsupported response mode {response_mode}")
        conversation = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": target_output},
        ]
        modified_example[f"{response_mode.value.lower()}_conversation"] = conversation
        modified_example[f"{response_mode.value.lower()}_target"] = target_output
    # del modified_example["decision"]
    return modified_example


def parse_outputs(example: Mapping[str, Any], problem: Problem) -> Mapping[str, Any]:
    example = dict(example)
    if "explanation_output" in example:
        try:
            explanation_decision_output = problem.parse_explanation_decision_output(
                example
            )
        except Exception as e:
            logging.warning(
                f"Error occurred while processing explanation output '{example['explanation_output']}': {e}"
            )
            explanation_decision_output = -1
        example["explanation_decision_output"] = explanation_decision_output
    if "decision_target" in example:
        example["decision_target"] = problem.parse_decision_target(example)
    if "decision_output" in example:
        try:
            decision_output = problem.parse_decision_output(example)
        except Exception as e:
            logging.warning(
                f"Error occurred while processing decision output '{example['decision_output']}': {e}"
            )
            decision_output = -1
        example["decision_output"] = decision_output
    if "reasoning_output" in example:
        reasoning_target = problem.parse_reasoning_target(example)
        try:
            reasoning_output = problem.parse_reasoning_output(example)
        except Exception as e:
            logging.warning(
                f"Error occurred while processing reasoning output '{example['reasoning_output']}': {e}"
            )
            reasoning_output = example["reasoning_output"]
        example["raw_reasoning_target"] = example["reasoning_target"]
        example["raw_reasoning_output"] = example["reasoning_output"]
        example["reasoning_target"] = reasoning_target
        example["reasoning_output"] = reasoning_output
    # example["raw_decision_target"] = example["decision_target"]
    # example["raw_decision_output"] = example["decision_output"]
    return example
