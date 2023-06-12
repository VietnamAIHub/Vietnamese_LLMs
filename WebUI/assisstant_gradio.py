'''
@TranNhiem 2023/06/06

This code is the first version of the LLM (Language Model) Assistant using the LangChain Tool. 

1. Loading The Finetuned Instruction LLM Model
    + Using Checkpoint 
    + Using HuggingFace Model Hub

2. Using The LangChain Tool 
    + LangChain Memory System (with Buffer Memory we no need openAI API)
    + LangChain VectorStore System (You Need OpenAI Embedding for this)

3. Connect to Vector Database (FAISS) for Indexing and Searching

4. Further Design LLMs for Auto Agent (AI Agent) LLM using Langchain Tool 

    4.1 Indexing LLM (Augmented Retrieved Documents )
    4.2 Agent LLM (Design & Invent New thing for Human)

5. Future work: Integrate Huggingface Text-generation-inference 
    + https://github.com/huggingface/text-generation-inference 
'''

import os
import torch
from langchain.memory import VectorStoreRetrieverMemory
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import os
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory,  ConversationBufferWindowMemory
from langchain.chains import ConversationChain 
from langchain import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.docstore import InMemoryDocstore

os.environ["OPENAI_API_KEY"] = ""

## -----------------------------------------------------------------
## Loading LLM Models (Loading from Checkpoint or HuggingFace Model Hub)
## -----------------------------------------------------------------
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
## Loading The FineTuned LoRa Adapter Model 
from peft import PeftModel, PeftConfig
import bitsandbytes as bnb
from transformers import  AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

## --------------------------Setting Update to Load Baseline Pretrained Model First---------------------------------------
base_model="bigscience/bloomz-1b7"
cache_dir="/content/bloomz/" # Path to save model weight to Disk

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    #cache_dir=cache_dir,
    # load_in_8bit=True, ## Currently RTX 1080Ti not working 
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model,  torch_dtype=torch.float16,cache_dir=cache_dir,)#

model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

## --------------------------Loading model from Checkpoint ---------------------------------------
def load_model_from_checkpoint(model, checkpoint_path="/content/drive/MyDrive/Generative_Model_Applications/checkpoint-48400/"):

    from_checkpoint=checkpoint_path
    checkpoint_name = os.path.join(from_checkpoint, "pytorch_model.bin")
    print(f"Restarting from {checkpoint_name}")
    adapters_weights = torch.load(checkpoint_name)
    for name, param in model.named_parameters():
        #if name == "base_model.model.transformer.h.1.self_attention.query_key_value.lora_A.default.weight":
        weight_tensor = adapters_weights[name]  # Get the corresponding tensor from weight_value
        param.data = weight_tensor  # Replace the parameter tensor with weight_tensor
    
    return model

## --------------------------Loading model from Checkpoint ---------------------------------------
def load_model_from_hub_or_local_path(model, model_name, model_path=None): 
   
    if model_path is not None:
        print("Loading Model from Local Path")
        model_name=model_path
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto', use_auth_token=True)#load_in_8bit=True
    else:
        print("Loading Model from HuggingFace Model Hub") 
        model=PeftModel.from_pretrained(model, model_name)
    
    return model 

checkpoint_path="/content/drive/MyDrive/Generative_Model_Applications/checkpoint-48400/"
model=load_model_from_checkpoint(model, checkpoint_path)
model.to('cuda')
## -----------------------------------------------------------------
## New Gradio WebAPP interface For New Feature and Advance interface 
## -----------------------------------------------------------------

_DEFAULT_TEMPLATE = """ Below is an instruction that describes a task. Please provide a response that appropriately completes the request, considering both the relevant information discussed in the ongoing conversation and disregarding any irrelevant details. If the AI does not know the answer to a question, it truthfully says it does not know.

### Current conversation:{history}

### Instruction: {input}

### Response:
"""

prompt_template_ = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)


import gradio as gr
with gr.Blocks() as demo:
    gr.Markdown("""<h1><center> SIF-LLM Assistant (Alpha Released)  </center></h1>""")
    
    with gr.Row(scale=4, min_width=300, min_height=100):
        with gr.Column():
          base_model_input = gr.Dropdown(choices= ["Alpha-7B1_Coming_Soon", "Alpha-1B7", ], value="Alpha-1B7", label="Choosing LLM", show_label=True)
        with gr.Column():
          conversation_style = gr.Dropdown( choices=["More Creative", "More Balance"], value="More Creative", label="Conversation Style", show_label=True)
                    
    chatbot = gr.Chatbot(label="Assistant").style(height=500)
    
    with gr.Row():
        message = gr.Textbox(show_label=False, placeholder="Enter your prompt and press enter", visible=True)
    state = gr.State()

  
    # max_token_limit=40 - token limits needs transformers installed
    #memory= ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)
    #memory= ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)

    def respond(message, chat_history,repetition_penalty=1.2, temperature=0.6, top_p=0.95, penalty_alpha=0.4,top_k=20, max_output_tokens=512, base_model='bloomz_1b7',conversation_style="More Creative"):
        
        ## Setup VectorStore Memory OR Buffer Memory 
        # embedding_size = 1536 #1536 # Dimensions of the OpenAIEmbeddings
        # index = faiss.IndexFlatL2(embedding_size)
        # embedding_fn = OpenAIEmbeddings().embed_query
        # vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        # retriever = vectorstore.as_retriever(search_kwargs=dict(k=4))
        # memory = VectorStoreRetrieverMemory(retriever=retriever)

        ## Most Simple one without require any Extra LLM for Embedding or Summary
        memory = ConversationBufferWindowMemory(k=4)



        ## Setting Up Pretrained Model 

       
        if conversation_style == "More Creative":
          pipe = pipeline(
              "text-generation",
              model=model, 
              tokenizer=tokenizer, 
              max_length=max_output_tokens,
              ## Contrastive Search Setting
              penalty_alpha=penalty_alpha, 
              top_k=top_k,
              repetition_penalty=repetition_penalty)

        else: 
          pipe = pipeline(
            "text-generation",
            model=model, 
            tokenizer=tokenizer, 
            max_length=max_output_tokens,
            ## Beam Search
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
            )
              
        
        local_llm = HuggingFacePipeline(pipeline=pipe)

        
        conversation_with_summary = ConversationChain(
            llm=local_llm, 
            memory=memory, 
            verbose=True, 
            prompt= prompt_template_, 
        )
        bot_message = conversation_with_summary.predict(input=message)
        message= "👤: "+ message
        bot_message= "Assistant 😃: "+ bot_message
        chat_history.append((message, bot_message))
        #time.sleep(1)
        return "", chat_history
     
     
    ## For Setting Hyperparameter 
    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        #It is a value between 1.0 and infinity, where 1.0 means no penalty
        repetition_penalty = gr.Slider(
            minimum=1.0,
            maximum=10.0,
            value=1.2,
            step=0.5,
            interactive=True,
            label="repetition_penalty",
        )
        
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        ## two values penalty_alpha & top_k use to set Contrastive Decoding for LLM 
        penalty_alpha = gr.Slider(
            minimum=0.001,
            maximum=1.0,
            value=0.4,# Values 0.0 mean equal to gready_search
            step=0.05,
            interactive=True,
            label="penalty_alpha",
        )
        top_k = gr.Slider(
            minimum=5.0,
            maximum=40.0,
            value=20,## Top number of candidates 
            step=2,
            interactive=True,
            label="Top_k",
        )
        max_output_tokens = gr.Slider(
            minimum=100,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    message.submit(respond, inputs=[message, chatbot, repetition_penalty,temperature, top_p, penalty_alpha, top_k, max_output_tokens,base_model_input, conversation_style], outputs=[message, chatbot], queue=False, )
    #gr.Interface(fn=respond, inputs=[message, chatbot, repetition_penalty, temperature, top_p, penalty_alpha, top_k, max_output_tokens, base_model_input, conversation_style], outputs=[message, chatbot], title="Alpha Assistant via SIF LLM", server_port=1234).launch(share=True)

demo.queue()
demo.launch(debug=True, server_port=1234, share=True)
