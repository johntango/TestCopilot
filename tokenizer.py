from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer_output = tokenizer.tokenize("This is an example of the bert tokenizer")
print(tokenizer_output)
# ['this', 'is', 'an', 'example', 'of', 'the', 'bert', 'token', '##izer']