from __init__ import *

delta = 1.0  # keep zone fixed

# Balanced 50/50
policy_bal, sigma_bal = kt.train_kto(BETA, delta, good_ratio=0.5)

# Imbalanced 10% Good / 90% Bad
policy_imbal, sigma_imbal = kt.train_kto(BETA, delta, good_ratio=0.1)

plt.figure()
plt.plot(sigma_bal, label="Balanced (50/50)")
plt.plot(sigma_imbal, label="Imbalanced (10/90)")
plt.xlabel("Training Step")
plt.ylabel("Sigma")
plt.title("Data Sensitivity Test")
plt.legend()
plt.savefig("images/KTO_data_sensitivity.png")
plt.show()