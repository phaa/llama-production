from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

model_id = "llava-hf/llava-1.5-7b-hf"  # escolha leve, pode mudar depois
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    #device_map="auto",
    load_in_4bit=True,  # Ou use load_in_8bit=True
    torch_dtype=torch.float16
).cuda()


async def analyze_image(image: Image.Image, prompt: str) -> str:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
    output = model.generate(**inputs, max_new_tokens=512)
    return processor.batch_decode(output, skip_special_tokens=True)[0]