'''
@@TranNhiem  2023/05

This code fine-tunes the LLM (Language Model) using the LoRa Alpaca pipeline.

Pipeline Overview:

1. LORA: Finetuned Model -> Further Compression to 8-Bit or 4-Bit Quantization -> FineTune Quantized Model
+   GPTQ Quantization: 
+ AWQ: Activation-aware Weight Quantization for LLM: https://github.com/mit-han-lab/llm-awq#awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration-paper

'''
'''
@TranNhiem 2023/06/09
For Finetuning Larger LLMs Model 


# 1. First Step: Download Directly from 
    Quantization GPTQ --> Quantized LLM Models to 4 Bits 
    https://github.com/IST-DASLab/gptq
    https://huggingface.co/blog/chatbot-amd-gpu 

# 2. Second Step: Using LoRA to FineTune LLM via Low Bit Percision 
    # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=s6f4z8EYmcJ6
    + https://huggingface.co/blog/4bit-transformers-bitsandbytes 

# 3. Further Optimization FineTuning via Deepspeed & Triton (Gradient Checkpointing) & Sparse LLMs
    + DeepSpeed Implementation

'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def setup_cache_directory(cache_dir):
    """
    Creates the cache directory if it does not exist.

    Args:
        cache_dir (str): The path to the cache directory.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

def setup_model(model_id, cache_dir):
    """
    Sets up the tokenizer and model.

    Args:
        model_id (str): The model identifier.
        cache_dir (str): The path to the cache directory.

    Returns:
        tokenizer: The tokenizer object.
        model: The model object.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, load_in_4bit=True, quantization_config=bnb_config, trust_remote_code=True)
    return tokenizer, model

def setup_peft_config(alpha, dropout, r):
    """
    Sets up the PeftConfig for the model.

    Args:
        alpha (int): The alpha value for LoRA.
        dropout (float): The dropout rate.
        r (int): The r value for LoRA.

    Returns:
        peft_config: The PeftConfig object.
    """
    peft_config = LoraConfig(
        lora_alpha=alpha,
        lora_dropout=dropout,
        r=r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "g_proj",
            "v_proj"
        ]
    )
    return peft_config

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: The model object.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def load_training_dataset(dataset_name, split):
    """
    Loads the training dataset.

    Args:
        dataset_name (str): The name of the dataset.
        split (str): The split of the dataset.

    Returns:
        dataset: The loaded dataset.
    """
    dataset = load_dataset(dataset_name, split=split)
    return dataset

def setup_training_arguments(output_dir, per_device_train_batch_size, gradient_accumulation_steps, optim, save_steps,
                            logging_steps, learning_rate, max_grad_norm, max_steps, warmup_ratio, group_by_length,
                            lr_scheduler_type, use_wandb, wandb_project, wandb_run_name, wandb_watch, wandb_log_model):
    """
    Sets up the training arguments for the trainer.

    Args:
        output_dir (str): The output directory for saving checkpoints and logs.
        per_device_train_batch_size (int): The batch size per device for training.
        gradient_accumulation_steps (int): The number of steps to accumulate gradients.
        optim (str): The optimizer to use.
        save_steps (int): The number of steps between saving checkpoints.
        logging_steps (int): The number of steps between logging training metrics.
        learning_rate (float): The learning rate for the optimizer.
        max_grad_norm (float): The maximum gradient norm for gradient clipping.
        max_steps (int): The maximum number of training steps.
        warmup_ratio (float): The warmup ratio for learning rate scheduling.
        group_by_length (bool): Whether to group samples by length during training.
        lr_scheduler_type (str): The type of learning rate scheduler to use.
        use_wandb (bool): Whether to use wandb for logging.
        wandb_project (str): The wandb project name.
        wandb_run_name (str): The wandb run name.
        wandb_watch (str): The wandb watch mode.
        wandb_log_model (str): The wandb log model mode.

    Returns:
        training_arguments: The TrainingArguments object.
    """
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )
    return training_arguments

def prepare_trainer(model, dataset, peft_config, tokenizer, max_seq_length, training_arguments):
    """
    Prepares the SFTTrainer for training.

    Args:
        model: The model object.
        dataset: The training dataset.
        peft_config: The PeftConfig object.
        tokenizer: The tokenizer object.
        max_seq_length (int): The maximum sequence length.
        training_arguments: The TrainingArguments object.

    Returns:
        trainer: The SFTTrainer object.
    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments
    )
    return trainer

def main():
    cache_dir = "/data/rick/pretrained_weights/LLaMA/"
    setup_cache_directory(cache_dir)

    model_id = "Neko-Institute-of-Science/LLaMA-7B-4bit-32g"
    tokenizer, model = setup_model(model_id, cache_dir)

    alpha = 16
    dropout = 0.1
    r = 64
    peft_config = setup_peft_config(alpha, dropout, r)
    model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    dataset_name = "timdettmers/openassistant-guanaco"
    dataset = load_training_dataset(dataset_name, split="train")

    output_dir = "./results"
    per_device_train_batch_size = 16
    gradient_accumulation_steps = 4
    optim = "adamw_bnb_8bit"
    save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 100
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"
    wandb_project = "Vietnamese_LLMs"
    wandb_run_name = "SFT_LLaMA_7B_QLORA_Alpaca_Vi"
    wandb_watch = "all"
    wandb_log_model = "true"

    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)

    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    training_arguments = setup_training_arguments(output_dir, per_device_train_batch_size,
                                                  gradient_accumulation_steps, optim, save_steps, logging_steps,
                                                  learning_rate, max_grad_norm, max_steps, warmup_ratio,
                                                  True, wandb_project, wandb_run_name, wandb_watch, wandb_log_model)

    max_seq_length = 2048
    trainer = prepare_trainer(model, dataset, peft_config, tokenizer, max_seq_length, training_arguments)

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()

if __name__ == "__main__":
    main()