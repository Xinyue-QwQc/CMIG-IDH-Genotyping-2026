from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch
from peft import get_peft_model, LoraConfig, PeftModel
# import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class M3D_langauage(nn.Module):
    def __init__(self,path="/home/cyx/Datasets/HuggingFace_models/M3D_CLIP/",use_lora=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            model_max_length=512,
            padding_side="right",
            use_fast=False
        )

        model = AutoModel.from_pretrained("/home/cyx/Datasets/HuggingFace_models/M3D_CLIP/", trust_remote_code=True)

        # 不使用这个会导致文本映射头也训练
        for n,p in model.named_parameters():
            p.requires_grad = False

        if use_lora:
            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=4,
                lora_alpha=4,
                lora_dropout=0.1,
                target_modules=["value","query","key"] # ,"key"
            )
            language_model = model.language_encoder
            language_model = get_peft_model(language_model,peft_config)
            model.language_encoder = language_model

        del model.vision_encoder

        self.model = model

        # if not use_lora:
        #     for p in self.model.parameters():
        #         p.requires_grad = False
    
    def forward_features(self,texts):

        with torch.no_grad():
            device_index = next(self.model.parameters()).device.index
            text_tensor = self.tokenizer(texts, max_length=512, padding="max_length", return_tensors="pt")
            input_id = text_tensor["input_ids"]
            attention_mask = text_tensor["attention_mask"]


            if device_index is not None:
                input_id = input_id.to(f"cuda:{device_index}")
                attention_mask = attention_mask.to(f"cuda:{device_index}")

            text_features = self.model.encode_text(input_id, attention_mask)[:, 0]
            return text_features


    def forward(self,text_prompts):

        global_features = self.forward_features(text_prompts)
        return global_features

        # ls = []
        # for text_prompt in text_prompts:
        #     sent_prompt = text_prompt.split("\n\n\n")
        #     ls.append(self.forward_features(sent_prompt))
        
        # local_features = torch.stack(ls,dim=0)

        # # return local_features

        # return torch.cat([global_features,local_features],dim=1)




        

        

        # text_tensor = self.tokenizer(text_prompts, max_length=512, padding="max_length", return_tensors="pt")
        # input_id = text_tensor["input_ids"]
        # attention_mask = text_tensor["attention_mask"]


        # if device_index is not None:
        #     input_id = input_id.to(f"cuda:{device_index}")
        #     attention_mask = attention_mask.to(f"cuda:{device_index}")

        # text_features = self.model.encode_text(input_id, attention_mask)[:, 0]
        # return text_features



if __name__ == "__main__":
    # word = "IDH-mutant gliomas are usually regular in shape with well-defined margins and usually occur in the frontal lobes, whereas IDH-wild-type gliomas are often poorly defined and associated with significant peritumoral edema."
    word = ('The Male brain glioma patient is 36 years old. \n\n\n Location:\nLeft parietal lobe, deep white matter, involving the subcortical region and extending towards the periventricular area.\n\n\nMorphology:\n- Well-defined, lobulated mass lesion\n- Irregular margins with infiltrative growth pattern\n- Significant mass effect causing compression of adjacent structures\n- Effacement of the fourth ventricle and mild midline shift\n\n\nMulti-modal Signal Characteristics:\nT1: Hypointense to isointense relative to gray matter.\nT1CE: Heterogeneous enhancement with irregular, nodular contrast uptake.\nT2: Hyperintense with a surrounding hypointense rim suggestive of vasogenic edema.\nFLAIR: Markedly hyperintense with extensive surrounding FLAIR signal abnormality indicative of extensive peritumoral edema.\n\n\nPeritumoral Findings:\nExtensive peritumoral edema is present, causing effacement of the adjacent sulci and gyri. The edema extends into the ipsilateral lateral ventricle, resulting in mild dilatation. The mass effect also leads to compression of the ipsilateral basal ganglia and midbrain structures. No evidence of hemorrhage or calcification within the tumor.', 'The Female brain glioma patient is 49 years old. \n\n\n Location:\nLarge, infiltrative right frontal lobe mass, involving the anterior and middle frontal gyri. The lesion extends into the right basal ganglia and insular cortex.\n\n\nMorphology:\nThe tumor demonstrates an irregular, lobulated contour with ill-defined margins. It causes significant mass effect, including compression of the right lateral ventricle and effacement of adjacent sulci. There is a midline shift of approximately 4 mm towards the left. The lesion appears to involve both gray and white matter components.\n\n\nMulti-modal Signal Characteristics:\nT1: The mass exhibits predominantly hypointense signal characteristics, with areas of iso- to mildly hyperintense signal within the central portion of the lesion.\nT1CE: There is heterogeneous enhancement following contrast administration, with prominent peripheral rim enhancement and some internal non-enhancing regions suggestive of necrosis or cystic degeneration.\nT2: The lesion demonstrates markedly hyperintense signal characteristics on T2-weighted sequences, consistent with high cellularity and associated vasogenic edema.\nFLAIR: Suppression of the surrounding hyperintense signal is observed, confirming the presence of extensive vasogenic edema surrounding the tumor.\n\n\nPeritumoral Findings:\nExtensive peritumoral vasogenic edema is present, causing significant mass effect and midline shift. The edema extends into the contralateral frontal lobe, compressing the left lateral ventricle. The basal ganglia, thalamus, and corpus callosum appear uninvolved, although there is some distortion secondary to the mass effect.', 'The Female brain glioma patient is 51 years old. \n\n\n Location:\nRight frontal lobe, extending to the right parietal lobe and corpus callosum.\n\n\nMorphology:\nIrregularly shaped mass with ill-defined margins. The lesion appears to involve both cortical and subcortical regions, with extension across midline structures (corpus callosum).\n\n\nMulti-modal Signal Characteristics:\nT1: Hypointense signal.\nT1CE: Irregular, heterogeneous enhancement with areas of non-enhancement suggestive of necrosis or cystic degeneration.\nT2: Heterogeneous hyperintensity.\nFLAIR: Suppression of CSF signal within the lesion, indicating involvement of white matter tracts.\n\n\nPeritumoral Findings:\nExtensive peritumoral vasogenic edema involving the right frontal lobe, extending into the corpus callosum. There is effacement of the adjacent sulci and gyri, as well as compression of the right lateral ventricle. The basal ganglia and thalamus on the right side appear displaced due to the mass effect.', 'The Male brain glioma patient is 43 years old. \n\n\n Location:\nRight frontal lobe, deep white matter, extending to the right basal ganglia and insular cortex.\n\n\nMorphology:\nIrregularly shaped mass with infiltrative margins. Marked surrounding edema causing effacement of adjacent sulci and gyri. Mass effect on the right lateral ventricle with midline shift towards the left.\n\n\nMulti-modal Signal Characteristics:\nT1: The lesion appears hypointense relative to gray matter.\nT1CE: There is heterogeneous enhancement with areas of intense contrast uptake as well as regions of non-enhancement suggestive of necrosis or cystic components.\nT2: The tumor is hyperintense on T2-weighted images, indicating increased water content.\nFLAIR: The lesion shows marked hyperintensity on FLAIR sequences, consistent with vasogenic edema surrounding the tumor.\n\n\nPeritumoral Findings:\nSignificant vasogenic edema surrounding the tumor, resulting in mass effect and compression of the adjacent brain structures. The ventricles appear to be compressed and displaced, likely due to the large size of the mass.')
    
    # words = ["The brain glioma Male patient is 36 years old.",
    #          "The brain glioma Female patient is 49 years old.",
    #          "The brain glioma Female patient is 51 years old.",
    #          "The brain glioma Male patient is 43 years old."
    #          ]*4
    # print(len(words))
    # print(words)
    with torch.no_grad():
        model = M3D_langauage()
        model = model
        text_f = model(word)
        # print(text_f.shape)
        torch.save(text_f,"BG_PT.pt")
    #     text_f = text_f.view(4,4,-1).permute(1,0,2)
    #     if torch.all(text_f[0][0] == text_f[0][3]):
    #         print("yes")
    #     else:
    #         print("no")
    # # msg = model.load_state_dict(torch.load("/home/cyx/Codes/SAM_MED3D_for_CLS/Language_check/params.pth"),strict=False)
    # # print(msg)
    # dic = {}
    # for n,p in model.named_parameters():
    #     if p.requires_grad:
    #         dic[n] = p
    #         print(n)
    # print(dic)
    # torch.save(dic,"/home/cyx/Codes/SAM_MED3D_for_CLS/Language_check/params.pth")
    # language_model = model.model.language_encoder
    # lora_model = PeftModel.from_pretrained(
    #     language_model,
    #     "/home/cyx/Codes/SAM_MED3D_for_CLS/Language_check"
    # )
    # print(lora_model)
    # language_model.save_pretrained("/home/cyx/Codes/SAM_MED3D_for_CLS/Language_check/")
    # print(model.model.language_encoder)
    # params = sum([param.nelement() for param in model.parameters()])/1e6
    # print(params)
    # text_prompt = "IDH-wildtype gliomas have blurred margins and a bithalamic rim enhancement around central necrosis."
    # text_f = model(text_prompt)
    # print(text_f)