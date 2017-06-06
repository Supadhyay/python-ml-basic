import numpy as np

np.random.seed(0)

totals = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}
purchases = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0}

totalPurchase = 0

for _ in range(100000):
    ageDecade = np.random.choice([20, 30, 40, 50, 60, 70])
    purchaseProbability = float(ageDecade) / 100.0
    totals[ageDecade] += 1
    if np.random.random() < purchaseProbability:
        totalPurchase += 1
        purchases[ageDecade] += 1


print("Totals dict : {}".format(totals))
print("Purchases dict : {}".format(purchases))
print("Total Purchases dict : {}".format(totalPurchase))


# Conditional Probability


# P(E|F) people in their 30's buying something

PEF = float(purchases[30]) / float(totals[30])
print("P(purchases | 30): {}".format(PEF))

# P(F) probability of being 30's

PF = float(totals[30]) / 100000.0
print("P(30's) : {}".format(PF))

# P(E) probability of buying something

PE = float(totalPurchase) / 100000.0
print("P(Purchases) : {}".format(PE))

# P(E)P(F)

print("P(30's) * P(Purchases) : {}".format(PE * PF))

# P(E, F)

print("P(30's , Purchases) : {}".format(purchases[30] / 100000.0))

# p(E|F) = P(E,F)/ P(F)

print("P(purchases  | 30) - computed with formula p(E|F) = P(E,F)/ P(F) : {}".format((purchases[30] / 100000.0) / PF))
