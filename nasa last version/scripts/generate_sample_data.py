"""
Generate sample input and output Excel files for demonstration
"""

import pandas as pd
import numpy as np

# sample exoplanet data (3 demo rows)
sample_data = {
    'koi_period': [10.5, 3.2, 87.4],
    'koi_time0bk': [134.5, 145.2, 156.8],
    'koi_impact': [0.3, 0.15, 0.8],
    'koi_duration': [5.2, 2.1, 9.8],
    'koi_depth': [150.0, 450.0, 89.5],
    'koi_prad': [1.1, 2.5, 0.8],
    'koi_teq': [550.0, 1200.0, 380.0],
    'koi_insol': [200.0, 850.0, 45.0],
    'koi_model_snr': [12.5, 25.3, 8.7],
    'koi_steff': [5700.0, 6200.0, 5100.0],
    'koi_slogg': [4.4, 4.2, 4.6],
    'koi_srad': [1.0, 1.3, 0.9],
    'koi_kepmag': [13.5, 12.1, 14.8]
}

# create input DataFrame
df_input = pd.DataFrame(sample_data)

# save input file
df_input.to_excel('data/sample_input.xlsx', index=False, engine='openpyxl')
print("✓ Created data/sample_input.xlsx")

# create output DataFrame with predictions (mock predictions)
df_output = df_input.copy()
df_output['predicted_label'] = ['CANDIDATE', 'CONFIRMED', 'FALSE_POSITIVE']
df_output['predicted_prob_FALSE_POSITIVE'] = [0.12, 0.08, 0.85]
df_output['predicted_prob_CANDIDATE'] = [0.75, 0.15, 0.10]
df_output['predicted_prob_CONFIRMED'] = [0.13, 0.77, 0.05]
df_output['row_id'] = [1, 2, 3]

# save output file
df_output.to_excel('data/sample_output.xlsx', index=False, engine='openpyxl')
print("✓ Created data/sample_output.xlsx")

print("\nSample data files created successfully!")
print(f"Input file: {len(df_input)} rows, {len(df_input.columns)} columns")
print(f"Output file: {len(df_output)} rows, {len(df_output.columns)} columns")
print("\nColumn names:")
print(", ".join(df_input.columns.tolist()))