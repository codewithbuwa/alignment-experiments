from __init__ import *

delta = 1.0  # keep zone fixed

# Balanced 50/50
w, l = build_dpo_dataset(good_ratio=.5)
policy_bal, sigma_bal = dp.train_dpo(beta = BETA, w = w, l = l)

# Imbalanced 10% Good / 90% Bad
w, l = build_dpo_dataset(good_ratio=.1)
policy_imbal, sigma_imbal = dp.train_dpo(beta = BETA, w = w, l = l)

plt.figure()
plt.plot(sigma_bal, label="Balanced (50/50)")
plt.plot(sigma_imbal, label="Imbalanced (10/90)", linestyle = "dashed")
plt.xlabel("Training Step")
plt.ylabel("Sigma")
plt.title("Data Sensitivity Test (DPO)")
plt.legend()
plt.savefig("images/DPO_data_sensitivity.png")
plt.show()