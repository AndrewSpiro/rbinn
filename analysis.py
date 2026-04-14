import pandas as pd
import io

def get_stats(data_input):
    column_names = ['model', 'clean', 'pgd_f', 'pgd_l', 'spsa_f', 'spsa_l', 'transfer', 'extra']
    
    df = pd.read_csv(io.StringIO(data_input), names=column_names, header=None)

    numeric_df = df[['clean', 'pgd_f', 'pgd_l', 'spsa_f', 'spsa_l', 'transfer']].copy()

    means = numeric_df.mean()
    stds = numeric_df.std()

    summary = pd.DataFrame({
        'Mean': means,
        'Std Dev': stds
    })
    
    summary['Report'] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(summary['Mean'], summary['Std Dev'])]
    
    return summary

csv_data = pd.read_csv('results.csv', header=None)
results = get_stats(csv_data)
print(results)