import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from typing import Literal
from peft import LoraConfig,get_peft_model,AutoPeftModel
# 假设像调用存储的预训练参数的话
# model = AutoPeftModel.from_pretrained(dir)

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BertModel(nn.Module):
    """
    Given age and gender information
    """
    def __init__(self,model_path="/home/cyx/Datasets/HuggingFace_models/models--michiyasunaga--BioLinkBERT-large/",
                 key: Literal['pooler_output', 'last_hidden_state'] = 'last_hidden_state'):
        super().__init__()
        self.key = key

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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

    def forward(self, texts):

        text_inputs = self.tokenizer(texts, return_tensors="pt",padding=True,truncation=True,max_length=20)
        device_index = next(self.text_encoder.parameters()).device.index


        if device_index is not None:
            text_inputs.to(f"cuda:{device_index}")

        outputs = self.text_encoder(**text_inputs)

        # if self.key == "pooler_output":
        #     text_embeddings = outputs.pooler_output
        # else:
        #     text_embeddings = outputs.last_hidden_state
        # text_embeddings = outputs.last_hidden_state
        t_global = outputs.pooler_output
        # text_embeddings.requires_grad = True

        return t_global  # text_embeddings,

if __name__ == "__main__":
    # texts = ('The Male brain glioma patient is 56.0 years old.',
    #          'The Male brain glioma patient is 30.0 years old.',
    #          'The Female brain glioma patient is 32.0 years old.',
    #          'The Male brain glioma patient is 43.46 years old.',
    #          'The Female brain glioma patient is 57.0 years old.',
    #          'The Male brain glioma patient is 51.0 years old.',
    #          'The Male brain glioma patient is 64 years old.',
    #          'The Male brain glioma patient is 40.0 years old.')
    texts = ("IDH-mutant gliomas are usually regular in shape with well-defined margins and usually occur in the frontal lobes, "
             "whereas IDH-wild-type gliomas are often poorly defined and associated with significant peritumoral edema.")
    model = BertModel(key="last_hidden_state").cuda()
    # print(model)
    out = model(texts)
    print(out.shape)
    # torch.save(out,"background.pt")
    # print(out.shape)
    # print("The",out.requires_grad)
    # # out.requires_grad = True
    #
    # for k in model.parameters():
    #     if k.requires_grad:
    #         print(k)
