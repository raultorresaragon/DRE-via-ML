import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

script_dir  = os.path.dirname(os.path.abspath(__file__))
images_dir  = os.path.join(script_dir, '../_1trt_effect/images')
os.makedirs(images_dir, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# LEFT PANEL - Propensity model
x1 = np.linspace(-4, 4, 1000)

y_logit = 1 / (1 + np.exp(-x1))
ax1.plot(x1, y_logit, linewidth=3, color='black', linestyle='-',  label='logit')

y_tanh = 0.5 * (np.tanh(x1) + 1)
ax1.plot(x1, y_tanh,  linewidth=3, color='black', linestyle='--', label='tanh')

ax1.set_xlabel(r'$x_i^T \beta_A$', fontsize=16)
ax1.set_ylabel('P(A=1)', fontsize=16)
ax1.set_title('Propensity model', fontsize=18)
ax1.legend(loc='lower right', fontsize=14)
ax1.tick_params(labelsize=14)
ax1.grid(True, alpha=0.3)

# RIGHT PANEL - Outcome model
x2 = np.linspace(-3, 3, 1000)
shape = 3
scale = 2

y_exp = np.exp(x2)
ax2.plot(x2, y_exp,      linewidth=3, color='black', linestyle='-',   label='exponential')

y_sigmoid = (1 / (1 + np.exp(-x2))) * 10
ax2.plot(x2, y_sigmoid,  linewidth=3, color='black', linestyle='--',  label='sigmoid')

y_loggamma = (np.exp(shape * x2) * np.exp(-np.exp(x2) / scale)) / (gamma(shape) * scale**shape) * 10
ax2.plot(x2, y_loggamma, linewidth=3, color='black', linestyle=':',   label='log-gamma')

y_lognormal = (1 / (np.exp(x2) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * x2**2) * 10
ax2.plot(x2, y_lognormal,linewidth=3, color='black', linestyle='-.',  label='lognormal')

ax2.set_xlabel(r'$x_i^T \beta_Y + \delta_1 A_i$', fontsize=16)
ax2.set_ylabel('Y', fontsize=16)
ax2.set_title('Outcome model', fontsize=18)
ax2.legend(loc='upper left', fontsize=14)
ax2.tick_params(labelsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'prop_outcome_funcs_bw.jpeg'), dpi=150, bbox_inches='tight')
plt.show()
