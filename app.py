# File: app.py
import streamlit as st
import random
import matplotlib.pyplot as plt
from openai import OpenAI
import os

# === CONFIGURE OPENAI ===
client = OpenAI(
    api_key="sk-proj-9_VOeHZnPpcJpGp3k4aGHWjo3-A-nwR4s7gStR74o0d5T-nRSH96JIw2UYY9POhlIdQGmQUVBnT3--jpjzc2NM_Z7eNGOIpdoLXSxSN0qUqf0BQk2oGGPaMgoihWvSKO49bOFC_ChFPwhrLqWN_ugA",
)  # or set directly: openai.api_key = "your-key"

# Constants
TARGET = 6174
POPULATION_SIZE = 10
MUTATION_RATE = 0.3
MAX_GENERATIONS = 20


# Kaprekar transformation
def kaprekar_step(n):
    digits = f"{n:04d}"
    high = int("".join(sorted(digits, reverse=True)))
    low = int("".join(sorted(digits)))
    return high - low


# Fitness function
def fitness(n):
    return -abs(n - TARGET)


# Mutation
def mutate(n):
    digits = list(f"{n:04d}")
    if random.random() < MUTATION_RATE:
        random.shuffle(digits)
    return int("".join(digits))


# Crossover
def crossover(a, b):
    da, db = list(f"{a:04d}"), list(f"{b:04d}")
    child = da[:2] + db[2:]
    return int("".join(child))


# Evolution
def evolve_population(pop):
    scored = sorted(pop, key=fitness, reverse=True)
    new_gen = scored[:2]  # Elitism
    while len(new_gen) < POPULATION_SIZE:
        parent1, parent2 = random.choices(scored[:5], k=2)
        child = mutate(crossover(parent1, parent2))
        child = kaprekar_step(child)
        new_gen.append(child)
    return new_gen


# GPT Commentary


def generate_ai_commentary(history):
    generations = len(history)
    snapshot = [f"Generation {i+1}: {history[i]}" for i in range(generations)]
    prompt = (
        "You're a convergence analyst. Provide a brief summary on the convergence pattern "
        "of the following evolutionary algorithm population data. Mention stability, rate of convergence, and interesting trends.\n\n"
        + "\n".join(snapshot)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data science assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to generate commentary: {e}"


# Streamlit App
st.title("🧬 Kaprekar-Inspired Genetic Algorithm")
st.write(
    "This simulation evolves a population of 4-digit numbers until they converge to 6174 using evolutionary heuristics."
)

# Sidebar controls
population_size = st.sidebar.slider("Population Size", 5, 30, POPULATION_SIZE)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, MUTATION_RATE, 0.05)
max_generations = st.sidebar.slider("Max Generations", 5, 100, MAX_GENERATIONS)

if st.button("🚀 Run Simulation"):
    POPULATION_SIZE = population_size
    MUTATION_RATE = mutation_rate
    MAX_GENERATIONS = max_generations

    population = [random.randint(1000, 9999) for _ in range(POPULATION_SIZE)]
    history = []

    for gen in range(1, MAX_GENERATIONS + 1):
        history.append(population[:])
        if all(p == TARGET for p in population):
            st.success(f"✅ Converged to {TARGET} in generation {gen}!")
            break
        population = evolve_population(population)
    else:
        st.warning("⚠️ Did not converge within the max number of generations.")

    # Visualization
    st.subheader("📊 Population Over Generations")
    fig, ax = plt.subplots()
    for i in range(POPULATION_SIZE):
        ax.plot(
            [gen[i] if i < len(gen) else None for gen in history], label=f"Genome {i}"
        )
    ax.axhline(y=TARGET, color="r", linestyle="--", label="Target 6174")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Genome Value")
    ax.set_title("Convergence of Population")
    st.pyplot(fig)

    # AI Commentary
    st.subheader("🧠 AI Commentary on Convergence")
    ai_comment = generate_ai_commentary(history)
    st.markdown(ai_comment)
