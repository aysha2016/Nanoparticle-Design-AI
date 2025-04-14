import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class AIOptimizer:
    def __init__(self):
        self.df = pd.DataFrame()
        self.model = RandomForestRegressor()

    def add_result(self, formulation, result):
        record = formulation.copy()
        record.update(result)
        self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)

    def train(self):
        if len(self.df) < 10:
            print("[AI] Not enough data to train model.")
            return
        X = self.df[["polymer_pct", "peg_da", "charge_mv"]]
        X = pd.get_dummies(pd.concat([X, self.df["ligand"]], axis=1))
        y = self.df["bbb_score"]
        self.model.fit(X, y)

    def suggest_next(self, base_formulation):
        new_formulation = base_formulation.copy()
        new_formulation["polymer_pct"] += 10 if new_formulation["polymer_pct"] < 70 else -10
        return new_formulation