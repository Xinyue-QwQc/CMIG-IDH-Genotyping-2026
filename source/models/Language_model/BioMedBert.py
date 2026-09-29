import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from typing import Literal
from peft import LoraConfig,get_peft_model,AutoPeftModel
# 假设像调用存储的预训练参数的话
# model = AutoPeftModel.from_pretrained(dir)

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class MedBertModel(nn.Module):
    """
    Given age and gender information
    """
    def __init__(self,model_path="/home/cyx/Datasets/HuggingFace_models/LLM_huggingface/medbert_pre/",
                 key: Literal['pooler_output', 'last_hidden_state'] = 'last_hidden_state', peft_path=None):
        super().__init__()
        self.key = key

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # if peft_path is not None:
        # text_encoder = AutoPeftModel.from_pretrained(peft_path)
        # else:
        text_encoder = AutoModel.from_pretrained(model_path)
        #     peft_config = LoraConfig(inference_mode=False, r=4, lora_alpha=1, lora_dropout=0)
        #     text_encoder = get_peft_model(text_encoder, peft_config)

        self.text_encoder = text_encoder
        # self.text_encoder.print_trainable_parameters()  get_peft_model(text_encoder,peft_config)


        for k in self.text_encoder.parameters():
            k.requires_grad = False

    def forward(self, text_inputs):

        # text_inputs = self.tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=20)
        # text_inputs.to(DEVICE)
        # for k,v in text_inputs.items():
        #     if isinstance(v,list):
        #         text_inputs[k] = torch.tensor(v)

        # with torch.no_grad():
        outputs = self.text_encoder(**text_inputs)

        # if self.key == "pooler_output":
        #     text_embeddings = outputs.pooler_output
        # else:
        #     text_embeddings = outputs.last_hidden_state
        text_embeddings = outputs.last_hidden_state
        t_global = outputs.pooler_output
        # text_embeddings.requires_grad = True

        return text_embeddings,t_global

if __name__ == "__main__":
    texts = ("IDH-wildtype gliomas have blurred margins and a bithalamic rim enhancement around central necrosis.",
             "IDH-wildtype gliomas have blurred margins and a bithalamic rim enhancement around central necrosis.")

    tokenizer = AutoTokenizer.from_pretrained("/home/cyx/Datasets/HuggingFace_models/LLM_huggingface/medbert_pre/")
    texts = tokenizer(texts)
    model = MedBertModel(key="last_hidden_state")
    outs = model(texts)
    print(outs)
    # for i in outs:
    #     print(i.shape)

    # print(model)
    # out = model(texts)
    # print(out.shape)
    # print("The",out.requires_grad)
    # # out.requires_grad = True
    #
    # for k in model.parameters():
    #     if k.requires_grad:
    #         print(k)
