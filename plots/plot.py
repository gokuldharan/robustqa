import pandas as pd
import matplotlib.pyplot as plt


baseline = pd.read_csv("run-baseline_oodomain-04-tag-val_F1.csv")
moe = pd.read_csv("run-MoE_04_oodomain-01-tag-val_F1.csv")

x_b, y_b = baseline['Step'].values, baseline['Value'].values
x_moe, y_moe = moe['Step'].values, moe['Value'].values



plt.plot(x_b, y_b, 'g', label='Baseline+oodomain')
plt.plot(x_moe, y_moe, 'b', label='MoE+oodomain')
plt.xlabel('Training Steps')
plt.ylabel('F1')
plt.legend()
plt.show()
