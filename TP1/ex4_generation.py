import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

MAX_LEN = 50  

print("SEED:", SEED)
print("PROMPT:", repr(prompt))
print("=" * 80)

print("\n=== (1) Greedy decoding ===")
out_greedy = model.generate(
    **inputs,
    max_length=MAX_LEN,
    do_sample=False,     
)
txt_greedy = tokenizer.decode(out_greedy[0], skip_special_tokens=True)
print(txt_greedy)

print("\nGreedy run x3 (should be identical):")
for i in range(3):
    out = model.generate(**inputs, max_length=MAX_LEN, do_sample=False)
    print(f"Run {i+1}:", tokenizer.decode(out[0], skip_special_tokens=True))

print("=" * 80)

def generate_once(seed, repetition_penalty=None, temperature=0.7):
    torch.manual_seed(seed)
    kwargs = dict(
        **inputs,
        max_length=MAX_LEN,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
    )
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty

    out = model.generate(**kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("\n=== (2) Sampling (temp=0.7, top_k=50, top_p=0.95), 5 generations ===")
for s in [1, 2, 3, 4, 5]:
    print("SEED", s)
    print(generate_once(s, temperature=0.7))
    print("-" * 60)

print("=" * 80)

print("\n=== (3) Repetition penalty comparison (seed controlled) ===")
seed_compare = 7
txt_no_pen = generate_once(seed_compare, repetition_penalty=None, temperature=0.7)
txt_pen = generate_once(seed_compare, repetition_penalty=2.0, temperature=0.7)

print("Seed:", seed_compare)
print("\n-- Without repetition_penalty --")
print(txt_no_pen)
print("\n-- With repetition_penalty=2.0 --")
print(txt_pen)

print("=" * 80)

print("\n=== (4) Temperature extremes (top_k=50, top_p=0.95) ===")
seed_temp = 9
txt_t01 = generate_once(seed_temp, temperature=0.1)
txt_t20 = generate_once(seed_temp, temperature=2.0)

print("Seed:", seed_temp)
print("\n-- temperature=0.1 --")
print(txt_t01)
print("\n-- temperature=2.0 --")
print(txt_t20)

print("=" * 80)

print("\n=== (5) Beam search (num_beams=5) ===")
out_beam5 = model.generate(
    **inputs,
    max_length=MAX_LEN,
    num_beams=5,
    do_sample=False,
    early_stopping=True
)
txt_beam5 = tokenizer.decode(out_beam5[0], skip_special_tokens=True)
print(txt_beam5)

print("=" * 80)

def timed_beam(num_beams: int, runs: int = 3):
    times = []
    last_text = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_length=MAX_LEN,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_text = tokenizer.decode(out[0], skip_special_tokens=True)

    return sum(times) / len(times), last_text

print("\n=== (6) Beam timing (avg over 3 runs) ===")
for b in [5, 10, 20]:
    avg_t, txt = timed_beam(b, runs=3)
    print(f"num_beams={b} -> avg_time={avg_t:.3f}s")
    print("Output:", txt)
    print("-" * 60)
