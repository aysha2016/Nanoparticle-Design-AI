import random

def generate_formulation():
    return {
        "polymer_pct": random.choice([40, 50, 60, 70]),
        "peg_da": random.choice([2000, 3000, 5000]),
        "ligand": random.choice(["TfR", "ApoE", "None"]),
        "charge_mv": random.choice([-15, -10, -5, 0]),
    }