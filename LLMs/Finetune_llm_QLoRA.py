'''
@@TranNhiem  2023/05

This code fine-tunes the LLM (Language Model) using the LoRa Alpaca pipeline.

Pipeline Overview:

1. LORA: Finetuned Model -> Further Compression to 8-Bit or 4-Bit Quantization -> FineTune Quantized Model
+   GPTQ Quantization: 
+ AWQ: Activation-aware Weight Quantization for LLM: https://github.com/mit-han-lab/llm-awq#awq-activation-aware-weight-quantization-for-llm-compression-and-acceleration-paper

'''
