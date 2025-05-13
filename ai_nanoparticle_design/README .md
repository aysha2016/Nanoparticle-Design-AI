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
   
   ```bash
   pip install pandas scikit-learn numpy
   ```

2. **Run the full pipeline**

   ```bash
   python main_loop.py
   ```

---

 

