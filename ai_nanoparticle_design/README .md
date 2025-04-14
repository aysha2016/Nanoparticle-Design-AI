# 🧪 Nanoparticle Optimization Closed-Loop System

This project simulates a **robot-assisted AI optimization loop** for generating, testing, and improving **nanoparticle formulations**. It mimics an experimental lab workflow using:

- A **formulation generator**
- An **assay simulator**
- A **robot controller**
- An **AI optimizer**
- A full **automated loop**

Perfect for rapid prototyping of smart lab automation tools!

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `formulation_generator.py` | Randomly generates nanoparticle formulations with polymer %, PEG size, surface ligand, and charge. |
| `assay_simulator.py` | Simulates experimental results: particle size, encapsulation efficiency, and blood-brain-barrier (BBB) penetration score. |
| `robot_controller.py` | Simulates sending formulation instructions to lab automation systems (e.g., Opentrons). |
| `ai_optimizer.py` | Implements a Random Forest model to learn from experimental data and suggest better formulations. |
| `main_loop.py` | Runs the full experiment loop: generates data, trains the AI, and iteratively improves results. |

---

## ⚙️ How It Works

1. **Initial Exploration**  
   - Generate 10 random formulations.
   - Simulate sending them to a robot.
   - Evaluate results using the assay simulator.

2. **AI Training**  
   - Train a `RandomForestRegressor` on the collected data.

3. **AI-Guided Optimization Loop**  
   - For 5 rounds, use the AI to suggest new formulations based on previous performance.
   - Simulate and log each result to refine the model.

---

## ▶️ Run the System

1. **Install dependencies**  
   *(Only if you're running the AI model training)*

   ```bash
   pip install pandas scikit-learn numpy
   ```

2. **Run the full pipeline**

   ```bash
   python main_loop.py
   ```

---

## 📈 Sample Output

```
[Robot] Executing formulation: {'polymer_pct': 60, 'peg_da': 2000, 'ligand': 'TfR', 'charge_mv': -5}
Simulated Results: size=105.2 nm, encapsulation=58.3%, BBB=0.78

[AI] Trained on 10 experiments.

--- Iteration 1 ---
[Robot] Executing formulation: {'polymer_pct': 70, 'peg_da': 2000, 'ligand': 'TfR', 'charge_mv': -5}
...
Final dataset:
   polymer_pct  peg_da ligand  charge_mv  size_nm  encapsulation_pct  bbb_score
...
```

---

## 🤖 Notes

- Real lab hardware (like Opentrons or microfluidic systems) can replace the `robot_controller` logic.
- Extend the AI model or reward strategy to optimize for multiple objectives (e.g., Pareto front).

---

## 🚀 Future Extensions

- Integrate Bayesian Optimization or Reinforcement Learning.
- Add real-world data logging & instrumentation.
- Replace simulators with real assay results.
