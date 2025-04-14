from formulation_generator import generate_formulation
from assay_simulator import simulate_assay
from robot_controller import send_to_robot
from ai_optimizer import AIOptimizer

optimizer = AIOptimizer()

# Initial random experiments
for _ in range(10):
    f = generate_formulation()
    send_to_robot(f)
    result = simulate_assay(f)
    optimizer.add_result(f, result)

# Train model
optimizer.train()

# AI-guided loop
base = generate_formulation()
for i in range(5):
    print(f"\n--- Iteration {i+1} ---")
    suggestion = optimizer.suggest_next(base)
    send_to_robot(suggestion)
    result = simulate_assay(suggestion)
    optimizer.add_result(suggestion, result)
    base = suggestion  # feedback loop

print("\nFinal dataset:")
print(optimizer.df.tail(10))