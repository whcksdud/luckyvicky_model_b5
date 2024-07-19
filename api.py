import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

model_id = "chan1121/luckyvicky2_model_5b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)

def gen(x):
    gened = model.generate(
        **tokenizer(
            f"### 질문: {x}\n\n### 답변:",
            return_tensors='pt',
            return_token_type_ids=False
        ),
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    tmp = tokenizer.decode(gened[0], skip_special_tokens=True).split('###')[2]
    end_token = "</끝>"
    token_position = tmp.find(end_token)
    if token_position != -1:
        result = tmp[:token_position]
    else:
        result = tmp
    result = result[4:]
    #print(result)
    return result

