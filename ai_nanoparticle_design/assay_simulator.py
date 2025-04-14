import numpy as np

def simulate_assay(formulation):
    size = 100 + np.random.randn() * 5 + (formulation["polymer_pct"] - 50) * 0.5
    encapsulation = 50 + (formulation["peg_da"] / 1000) * 5 + np.random.randn() * 2
    bbb_score = 0.5 + 0.05 * (formulation["polymer_pct"] / 10) \
                      - 0.02 * abs(formulation["charge_mv"]) \
                      + (0.1 if formulation["ligand"] == "TfR" else 0.05 if formulation["ligand"] == "ApoE" else 0)

    return {
        "size_nm": round(size, 2),
        "encapsulation_pct": round(encapsulation, 2),
        "bbb_score": round(min(max(bbb_score, 0), 1), 2)
    }