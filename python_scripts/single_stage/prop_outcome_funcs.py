import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Set up the figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Define colors
blue = '#3266ad'
teal = '#1D9E75'
coral = '#D85A30'
amber = '#BA7517'

# LEFT PANEL - Propensity model
x1 = np.linspace(-4, 4, 1000)

# Logistic (logit)
y_logit = 1 / (1 + np.exp(-x1))
ax1.plot(x1, y_logit, linewidth=4, color=blue, label='logit')

# Tanh
y_tanh = 0.5 * (np.tanh(x1) + 1)
ax1.plot(x1, y_tanh, linewidth=4, color=teal, label='tanh')

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

# Exponential
y_exp = np.exp(x2)
ax2.plot(x2, y_exp, linewidth=4, color=blue, label='exponential')

# Sigmoid (scaled by 10)
y_sigmoid = (1 / (1 + np.exp(-x2))) * 10
ax2.plot(x2, y_sigmoid, linewidth=4, color=teal, label='sigmoid')

# Log-gamma (scaled by 10)
y_loggamma = (np.exp(shape * x2) * np.exp(-np.exp(x2) / scale)) / (gamma(shape) * scale**shape) * 10
ax2.plot(x2, y_loggamma, linewidth=4, color=coral, label='log-gamma')

# Lognormal (scaled by 10)
y_lognormal = (1 / (np.exp(x2) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * x2**2) * 10
ax2.plot(x2, y_lognormal, linewidth=4, color=amber, label='lognormal')

ax2.set_xlabel(r'$x_i^T \beta_Y + \delta_1 A_i$', fontsize=16)
ax2.set_ylabel('Y', fontsize=16)
ax2.set_title('Outcome model', fontsize=18)
ax2.legend(loc='upper left', fontsize=14)
ax2.tick_params(labelsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/_1trt_effect/images/prop_outcome_funcs.jpeg', dpi=150, bbox_inches='tight')
plt.show()