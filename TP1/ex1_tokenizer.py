from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


phrase = "Artificial intelligence is metamorphosing the world!"

tokens = tokenizer.tokenize(phrase)

print("Phrase :", phrase)
print("Tokens :", tokens)

print("\n" + "="*80 + "\n")

token_ids = tokenizer.encode(phrase, add_special_tokens=False)

print("Token IDs :", token_ids)
print("\nDétails par token :")

for tid in token_ids:
    decoded = tokenizer.decode([tid])
    print(tid, repr(decoded))

print("\n" + "="*80 + "\n")

phrase2 = (
    "GPT models use BPE tokenization to process unusual words "
    "like antidisestablishmentarianism."
)

tokens2 = tokenizer.tokenize(phrase2)

print("Phrase 2 :", phrase2)
print("Tokens phrase 2 :", tokens2)

long_word = "antidisestablishmentarianism"
subtokens_long = [t for t in tokens2 if "anti" in t or "establish" in t or "arian" in t]

print("\nSous-tokens du mot long (approximation) :")
print(subtokens_long)
print("Nombre de sous-tokens :", len(subtokens_long))
