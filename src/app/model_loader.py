import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from functools import lru_cache

#MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

@lru_cache(maxsize=1)
def get_model_and_processor():
    print("Carregando modelo...")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    print("Modelo carregado com sucesso!")
    return processor, model