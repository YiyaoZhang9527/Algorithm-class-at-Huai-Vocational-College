documents = [
"太阳能电池板是一种可再生能源，对环境有益。",
"风力涡轮机利用风能发电。",
"地热供暖利用来自地球的热量为建筑物供暖。",
"水电是一种可持续能源，依靠水流发电。",
# 根据需要添加更多文档
]


from transformers import BertTokenizer, BertModel
import torch
 
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
 
# Tokenize and encode the documents
document_embeddings = []
for document in documents:
    inputs = tokenizer(document, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    document_embedding = outputs.last_hidden_state.mean(dim=1)  # Average over tokens
    document_embeddings.append(document_embedding)
document_embeddings = torch.cat(document_embeddings)