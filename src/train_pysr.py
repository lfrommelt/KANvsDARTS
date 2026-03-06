from pysr import PySRRegressor


def train_pysr(config, dataset, seed=42):
    model = PySRRegressor(**config)
    model.fit(dataset["train_input"], dataset["train_label"])

    sorted_df = model.equations.sort_values(
        "loss", ascending=True
    )  # .loc[0, "sympy_format"]
    sorted_df = sorted_df.reset_index()

    top5 = [
        (
            sorted_df.loc[i, "sympy_format"].simplify(),
            sorted_df.loc[i, "loss"],
            sorted_df.loc[i, "score"],
        )
        for i in range(min(5, len(sorted_df)))
    ]

    return top5
