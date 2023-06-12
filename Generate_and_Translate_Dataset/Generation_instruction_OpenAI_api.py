'''
@@TranNhiem 2023/06

This code uses the OpenAI API (gpt-3.5-turbo) & GPT-4 API for generate the self-instruction dataset for LLMs in Target Language (Vietnamese or Other Language)

## Reference Design: 
    + 1st Design: Using the approach from Standford Alpaca 175 Initial Instruction Human Created Dataset 
    Ref: Standford Alpaca: https://crfm.stanford.edu/2023/03/13/alpaca.html 
    + 2nd Approach: Using the approach 110K high-quailty user-based instructions from Discord 
    Ref: Instruction Wild: https://github.com/XueFuzhao/InstructionWild 
    + synthetic GPT-4 outputs GPTeacher, the general, roleplay v1&2, code instruct datasets, Nous Instruct & PDACTL (unpublished), CodeAlpaca, Evol_Instruct Uncensored, GPT4-LLM,

'''