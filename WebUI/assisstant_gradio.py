'''
@TranNhiem 2023/05

This design including 2 Sections:

1. Using The Open API LLM Model 
    + OpenAI API (gpt-3.5-turbo) & GPT-3 API (text-davinci-003)

1.. FineTune Instruction LLM  --> 2.. Langain Memory System  --> Specific Design Application Domain 

    4.1 Indexing LLM (Augmented Retrieved Documents )
    4.2 Agent LLM (Design & Invent New thing for Human)

'''

from langchain.memory import VectorStoreRetrieverMemory
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
import os
from langchain.llms import HuggingFacePipeline
from langchain import LLMChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain 
from langchain import OpenAI
from langchain.prompts.prompt import PromptTemplate


os.environ["OPENAI_API_KEY"] = ""

from langchain.docstore import InMemoryDocstore



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
model.to('cuda')

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
        
        ## Setup VectorStore Memory
        embedding_size = 1536 #1536 # Dimensions of the OpenAIEmbeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=4))
        memory = VectorStoreRetrieverMemory(retriever=retriever)

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
