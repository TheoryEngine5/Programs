import random

def generate_snippet():
    ops = ["+", "-", "*", "//", "**", "%"]
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    op = random.choice(ops)
    return f"{a} {op} {b}"

def safe_eval(expression):
    """Safely evaluate expression and handle potential errors"""
    try:
        result = eval(expression)
        # Handle division by zero or other edge cases
        if isinstance(result, (int, float)) and not (result != result):  # Check for NaN
            return result
        return 0
    except (ZeroDivisionError, OverflowError, ValueError):
        return 0

def mutate_snippet(snippet):
    """Create a mutated version of an existing snippet"""
    parts = snippet.split()
    if len(parts) == 3:
        a, op, b = parts
        # Randomly mutate one component
        mutation_type = random.choice(['a', 'op', 'b'])
        
        if mutation_type == 'a':
            a = str(random.randint(1, 10))
        elif mutation_type == 'op':
            ops = ["+", "-", "*", "//", "**", "%"]
            op = random.choice(ops)
        else:  # b
            b = str(random.randint(1, 10))
        
        return f"{a} {op} {b}"
    return generate_snippet()

def evolve_snippets(generations=5, population_size=20, keep_top=5):
    """Evolve snippets over multiple generations"""
    
    # Initialize population
    population = []
    for _ in range(population_size):
        snippet = generate_snippet()
        score = safe_eval(snippet)
        population.append((snippet, score))
    
    print(f"Starting evolution with {population_size} snippets over {generations} generations...")
    print("=" * 60)
    
    for gen in range(generations):
        # Sort by score (descending)
        population.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Generation {gen + 1}:")
        print(f"Best: {population[0][0]} -> {population[0][1]}")
        print(f"Avg:  {sum(p[1] for p in population) / len(population):.2f}")
        
        # Keep top performers
        survivors = population[:keep_top]
        
        # Generate new population
        new_population = survivors.copy()
        
        # Fill rest with mutations of top performers
        while len(new_population) < population_size:
            parent = random.choice(survivors[:3])  # Select from top 3
            mutated = mutate_snippet(parent[0])
            score = safe_eval(mutated)
            new_population.append((mutated, score))
        
        population = new_population
        print("-" * 40)
    
    # Final results
    population.sort(key=lambda x: x[1], reverse=True)
    print("\nFinal Top 3:")
    for i, (snippet, score) in enumerate(population[:3], 1):
        print(f"{i}. {snippet} -> Score: {score}")
    
    return population[:3]

# Run the original simple version
print("Simple Random Generation:")
print("=" * 30)
history = []
for _ in range(10):
    snippet = generate_snippet()
    score = safe_eval(snippet)
    history.append((snippet, score))

history.sort(key=lambda x: x[1], reverse=True)
for s, score in history[:3]:
    print(f"Top snippet: {s} -> Score: {score}")

print("\n" + "=" * 60)
print()

# Run the evolutionary version
evolve_snippets()