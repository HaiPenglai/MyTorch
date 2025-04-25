from sklearn.datasets import load_files
import torch
from transformers import DistilBertForSequenceClassification
from safetensors.torch import load_file
from mytorch.mynn import DistilBertConfig, CustomDistilBert

if __name__ == "__main__":
    input_ids = torch.randint(0, 30000, (1, 512))  # 示例输入
    attention_mask = torch.ones(1, 512)             # 示例attention mask
    
    config = DistilBertConfig()
    model = CustomDistilBert(config)
    for name, param in model.named_parameters():
        print(name, param.shape)
    model.load_state_dict(load_file("/home/dyx/nlp/Text-Classification/cc_fakenews/model_hub/bert.safetensors"))    
    outputs = model(input_ids, attention_mask=attention_mask)
    print("output", outputs)    
        
        
    official_model = DistilBertForSequenceClassification.from_pretrained("/home/dyx/nlp/_model/distilbert-base-chinese/distilbert-base-chinese", num_labels=2)
    official_model.load_state_dict(load_file("/home/dyx/nlp/Text-Classification/cc_fakenews/model_hub/bert.safetensors"))
    for name, param in official_model.named_parameters():
        print(name, param.shape)
    official_outputs = official_model(input_ids, attention_mask=attention_mask)
    print(official_outputs)








