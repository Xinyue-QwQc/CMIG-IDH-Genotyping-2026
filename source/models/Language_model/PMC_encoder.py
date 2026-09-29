import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Literal

class PMC_text_encoder(nn.Module):
    def __init__(self,model_path="/home/cyx/Datasets/HuggingFace_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/",
                 pmc_encoder="/home/cyx/Datasets/PMC_CLIP_checkpoint.pt",
                 key: Literal['pooler_output', 'last_hidden_state'] = 'pooler_output',):
        super().__init__()
        self.key = key

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_encoder = AutoModel.from_pretrained(model_path)

        # 导入PMC_CLIP的text encoder参数
        check = torch.load(pmc_encoder, map_location="cpu")["state_dict"]
        need_text_dict = {}
        for k, v in check.items():
            if "text_encoder" in k:
                need_text_dict[k[len("module.text_encoder."):]] = check[k]

        msg = self.text_encoder.load_state_dict(need_text_dict,strict=False)

        for k in self.text_encoder.parameters():
            k.requires_grad = False

    def forward(self, text_inputs):

        # with torch.no_grad():
        outputs = self.text_encoder(**text_inputs)

        # if self.key == "pooler_output":
        #     text_embeddings = outputs.pooler_output
        # else:
        #     text_embeddings = outputs.last_hidden_state

        # text_embeddings.requires_grad = True
        text_embeddings = outputs.last_hidden_state
        t_global = outputs.pooler_output

        return text_embeddings,t_global

if __name__ == "__main__":

    IDH_mutant = "IDH-mutant gliomas have clear tumor edges and tiny cysts."
    IDH_wild = " IDH-wildtype gliomas have blurred margins and a bithalamic rim enhancement around central necrosis."


    # texts = ('The Male brain glioma patient is 56.0 years old.',
    #          'The Male brain glioma patient is 30.0 years old.',
    #          'The Female brain glioma patient is 32.0 years old.',
    #          'The Male brain glioma patient is 43.46 years old.',
    #          'The Female brain glioma patient is 57.0 years old.',
    #          'The Male brain glioma patient is 51.0 years old.',
    #          'The Male brain glioma patient is 64 years old.',
    #          'The Male brain glioma patient is 40.0 years old.')

    model = PMC_text_encoder(key="pooler_output")
    tokenizer = AutoTokenizer.from_pretrained("/home/cyx/Datasets/HuggingFace_models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/")
    inputs = tokenizer(IDH_wild,return_tensors="pt",padding=True,truncation=True,max_length=77)
    out = model(inputs)
    print(out.shape)
    # torch.save(out,"/home/cyx/Datasets/IDH_wild.pt",)
