'''
@@TranNhiem  2023/05

This code fine-tunes the LLM (Language Model) using the LoRa Alpaca pipeline.
Code Reference from: https://github.com/tloen/alpaca-lora 

1. Using Open-Source Pretrained Language Model 
    + BLOOMZ 
    + LLaMA
    + Redpajama 
    + MPT 


2. Supervised Self-Instruct Finetune Model on Different Dataset via Vietnamese Version
    + Alpaca Instruction Style 
    + Share GPT Conversation Style 
    + Domain Target Instruction Style 


3. Optimization Pipeline Finetuning and Training:
    + Huggingface PEFT
        LORA: Finetuned Model -> 
    + Further Optimize via (Deepspeed, Colossal AI) for Training and Tunning LLMs

'''

import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

import json 
import os.path as osp
from typing import Union

## This prompter working for Alpaca Instruction dataset Structure
class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False, template_json_path: str = ""):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
            raise ValueError("Other Dataset is not supported yet")
        #file_name = osp.join("templates", f"{template_name}.json")
        file_name= template_json_path

        print(f'This is file name {file_name}')
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


###------------------------------------------------
## LORA Techniques  
###------------------------------------------------

## The available Implementation LORA Configure for LLM
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING ={
"t5": ["q", "v"],
"mt5": ["q", "v"], # Multi-lingual T5
"bart": ["q_proj","v_proj"],
"gpt2": ["c_attn"],
"bloom": ["query_key_value"],
"opt": ["g_proj", "v_proj"],
"gptj": ["q_proj", "v_proj"],
"gpt_neox": ["query_key_value"],
"gpt-neo": ["a_pros", "v_proj"],
"bert": ["query", "value"],
"roberta": ["query", "value"],
"xlm-roberta": ["query", "value"],
"electra": ["query", "value"],
"deberta-v2": ["query_proj","value_proj"],
"deberta": ["in _proj"],
"layoutlm": ["query","value"],
"llama": ["g_proj", "v _proj"],
"chatglm": ["query_key_value"],
}



def train(
    # model/data params
    base_model: str = "ckip-joint/bloom-1b1-zh", #"facebook/opt-350m",  # the only required argument
    cache_dir: str ="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/BLOOMZ/", #"/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/pretrained_weights/OPT/",
    data_path:  str="/media/rick/f7a9be3d-25cd-45d6-b503-7cb8bd32dbd5/Instruction_finetune_dataset/converted_Traditional_chinese_Belle_open_source_0_5M.json",
    #data_path: str =  "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json",#"/data/rick/Instruction_finetune_dataset/converted_Traditional_chinese_Belle_open_source_0_5M.json",
    output_dir: str = "./lora-alpaca_BlOOMZ_1b7m_0_5M_Traditional_CN/",
    template_json_path= "/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/data_structure_template/alpaca.json", #"/data/rick/LLM/Multimodal_Integrated_App/Language/data/data_structure_template/alpaca.json",
    # training hyperparams
    num_gpus=8,
    batch_size: int = 240,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 400,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,

    ## Configure Optimize HARDWARE memory 
    deepspeed_configure="", 
    optimizer_type="", 

    ## Depend on Different Model Architecutre setting this different
    lora_target_modules: List[str] = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["bloom"],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "Instructed_finetune_LLM",
    wandb_run_name: str = "Instruction_Finetune_BLOOM_1b7m_LORA_0_5M_Tradition_CN",
    wandb_watch: str = "all",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    resume_from_checkpoint: str = None, #"/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/lora-alpaca_BlOOMZ_1b7m_0_5M_Traditional_CN/checkpoint-48200/",  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    ## Create the path to store the weight download from HuggingFace Hub
    if not os.path.exists(cache_dir): 
        os.makedirs(cache_dir)

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name,template_json_path=template_json_path)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        # load_in_8bit=True, ## Currently RTX 2080Ti not support 8bit Opitmizer
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir, torch_dtype=torch.float16)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )
            resume_from_checkpoint = False

        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()


    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    '''
    Consideration implement DeepSpeed 
        Optimizer state partitioning (ZeRO stage 1)
        Gradient partitioning (ZeRO stage 2)
        Parameter partitioning (ZeRO stage 3)
        Custom mixed precision training handling
        A range of fast CUDA-extension-based optimizers
        ZeRO-Offload to CPU and NVMe

    '''

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            #fsdp= "full_shard auto_wrap offload",
            deepspeed="/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/deep_speed_configure/deep_speed_stage_3.json", #"/data/rick/LLM/Multimodal_Integrated_App/Language/deep_speed_configure/deep_speed_stage_3.json",
            #optim=bnb.optim.Adam8bit(), #"adamw_torch",
            optim="adamw_bnb_8bit",#, ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad']
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print("\nIf there's a warning about missing keys above, please disregard :)")

if __name__ == "__main__":
    # fire.Fire(train)
    train()