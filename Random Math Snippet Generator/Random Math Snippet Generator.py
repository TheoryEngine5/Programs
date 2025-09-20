import random

def generate_snippet():
    ops = ["+", "-", "*", "//", "**"]
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    op = random.choice(ops)
    return f"{a} {op} {b}"

history = []
for i in range(10):
    snippet = generate_snippet()
    score = eval(snippet)  # naive performance metric
    history.append((snippet, score))

# Keep top 3
history.sort(key=lambda x: x[1], reverse=True)
for s, score in history[:3]:
    print(f"Top snippet: {s} -> Score: {score}")