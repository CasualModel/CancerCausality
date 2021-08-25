import pandas as pd


# Define Concavity Dispersion Extraction Method


def calculate_concavity_dispersion(concave_points: pd.Series, area_val: pd.Series) -> list:
    values = []
    for index, value in concave_points.items():
        values.append(round(value / area_val[index], 6))

    return values
