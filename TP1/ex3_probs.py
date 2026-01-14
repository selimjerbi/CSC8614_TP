import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def token_probs_for_sentence(model, tokenizer, sentence: str):
    """
    Retourne pour chaque token t>=1 : P(x_t | x_<t) et logP(x_t | x_<t),
    + total_logp, perplexity.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0]  

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits 

    probs = torch.softmax(logits, dim=-1)             
    log_probs = torch.log_softmax(logits, dim=-1)    

    per_token = []
    total_logp = 0.0
    n = 0

    for t in range(1, len(input_ids)):
        tok_id = input_ids[t].item()
        p = probs[0, t - 1, tok_id].item()
        lp = log_probs[0, t - 1, tok_id].item()
        tok_txt = tokenizer.decode([tok_id])

        per_token.append((t, tok_txt, tok_id, p, lp))
        total_logp += lp
        n += 1

    avg_neg_logp = - total_logp / n
    ppl = math.exp(avg_neg_logp)

    return per_token, total_logp, ppl, input_ids

def print_token_probs(per_token, max_lines=None):
    """
    Affiche un extrait: t, token, proba.
    """
    if max_lines is None:
        max_lines = len(per_token)
    for (t, tok_txt, tok_id, p, lp) in per_token[:max_lines]:
        print(t, repr(tok_txt), f"P={p:.3e}", f"logP={lp:.3f}")

def main():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    phrase = "Artificial intelligence is fascinating."
    per_token, total_logp, ppl, input_ids = token_probs_for_sentence(model, tokenizer, phrase)

    print("=== (1) Token probs for:", repr(phrase))
    print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids.tolist()))
    print("\nExtrait (t, token, P, logP):")
    print_token_probs(per_token, max_lines=50)

    print("\n=== (2) Total log-prob and perplexity")
    print("total_logp:", total_logp)
    avg_neg_logp = - total_logp / max(1, len(per_token))
    print("avg_neg_logp:", avg_neg_logp)
    print("perplexity:", ppl)

    s1 = "Artificial intelligence is fascinating."
    s2 = "Artificial fascinating intelligence is."

    _, logp1, ppl1, _ = token_probs_for_sentence(model, tokenizer, s1)
    _, logp2, ppl2, _ = token_probs_for_sentence(model, tokenizer, s2)

    print("\n=== (3) Compare English sentences")
    print("S1:", repr(s1))
    print("  total_logp:", logp1)
    print("  ppl:", ppl1)
    print("S2:", repr(s2))
    print("  total_logp:", logp2)
    print("  ppl:", ppl2)

    fr = "L'intelligence artificielle est fascinante."
    _, logp_fr, ppl_fr, _ = token_probs_for_sentence(model, tokenizer, fr)

    print("\n=== (4) French sentence")
    print("FR:", repr(fr))
    print("  total_logp:", logp_fr)
    print("  ppl:", ppl_fr)

    prefix = "Artificial intelligence is"
    inp = tokenizer(prefix, return_tensors="pt")

    with torch.no_grad():
        out = model(**inp)
        logits2 = out.logits  

    last_index = logits2.shape[1] - 1
    last_logits = logits2[0, last_index, :]  
    last_probs = torch.softmax(last_logits, dim=-1)

    topk = 10
    vals, idx = torch.topk(last_probs, k=topk)

    print("\n=== (5) Top-10 next tokens after prefix:", repr(prefix))
    for p, tid in zip(vals.tolist(), idx.tolist()):
        print(repr(tokenizer.decode([tid])), f"{p:.3e}")

if __name__ == "__main__":
    main()
