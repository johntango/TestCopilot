from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("This is an example of the bert tokenizer")
print(tokens)
# ['this', 'is', 'an', 'example', 'of', 'the', 'bert', 'token', '##izer']

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
# [2023, 2003, 2019, 2742, 1997, 1996, 14324, 19204, 17629]

token_ids = tokenizer.encode("This is an example of the bert tokenizer")
print(token_ids)
# [101, 2023, 2003, 2019, 2742, 1997, 1996, 14324, 19204, 17629, 102]

tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
# ['[CLS]', 'this', 'is', 'an', 'example', 'of', 'the', 'bert', 'token', '##izer', '[SEP]']

import torch
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")

# get the embedding vector for the word "example"
example_token_id = tokenizer.convert_tokens_to_ids(["example"])[0]
example_embedding = model.embeddings.word_embeddings(torch.tensor([example_token_id]))

print(example_embedding.shape)
# torch.Size([1, 768])
print(example_embedding)

def get_embedding(word):
    token_id = tokenizer.convert_tokens_to_ids([word])[0]
    embedding = model.embeddings.word_embeddings(torch.tensor([token_id]))
    return embedding

def get_similar_words(word, num_similar=5):
    embedding = get_embedding(word)
    cos = torch.nn.CosineSimilarity(dim=1)
    cosines = []
    for i in range(tokenizer.vocab_size):
        cosine = cos(embedding, get_embedding(tokenizer.convert_ids_to_tokens([i])[0]))
        cosines.append((cosine, tokenizer.convert_ids_to_tokens([i])[0]))
    cosines.sort(reverse=True)
    return cosines[:num_similar]

def print_similarity_between_words(word1, word2):
    embedding1 = get_embedding(word1)
    embedding2 = get_embedding(word2)
    cos = torch.nn.CosineSimilarity(dim=1)
    similarity = cos(embedding1, embedding2)
    # print similarity between word1 and word2
    print(similarity.item())

# use f to print two words

print_similarity_between_words("king", "queen")
# 0.6510952
