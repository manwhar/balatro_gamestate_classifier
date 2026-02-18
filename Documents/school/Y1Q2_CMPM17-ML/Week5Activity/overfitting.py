import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend import best_fit, x, y_test, y_train

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# --------------------------------------------------------------------------------------------------------

# INSTRUCTIONS: Plot the training data in the first subplot (0, 0)
# Hint: everything uses the same x values
# Hint: Always plot training data as a consistent color (eg. blue)
axs[0, 0].scatter(x, y_train)
axs[0, 0].set_title("Training Data")

# --------------------------------------------------------------------------------------------------------

# INSTRUCTIONS: Plot the test data in the second subplot (0, 1)
# Hint: Always plot test data as a DIFFERENT consistent color (eg. red)
axs[0, 1].scatter(x, y_test)
axs[0, 1].set_title("Test Data")

# --------------------------------------------------------------------------------------------------------

# We've given you the code to get a best fit line for different complexities
# INSTRUCTIONS: Try plotting the best fit line for complexity=1 and the training data in the third subplot (1, 0)
# Hint: Plot best fit as a line (not scatter)
axs[1, 0].scatter(x, y_train)
y_fit = best_fit(1)
axs[1, 0].plot(x, y_fit)
axs[1, 0].set_title("Best Fit Line (Complexity 1)")

# --------------------------------------------------------------------------------------------------------

# Try plotting the training data and the best fit for complexity=20 in the last subplot (1, 1)
# Hint: use y_fit = best_fit(complexity) to get the y values for the best fit line
# complexity = 7
axs[1, 1].scatter(x, y_train)
complexities = [best_fit(i) for i in range(20)]
# y_fit = best_fit(complexity)
axs[1, 1].set_title(f"Best Fit Line (All complexities)")
[axs[1, 1].plot(x, i, alpha=0.3) for i in complexities]

# --------------------------------------------------------------------------------------------------------
# INSTRUCTIONS: Go back to the third (1, 0) and fourth (1, 1) subplots and plot the test data instead of the training data

# --------------------------------------------------------------------------------------------------------

# INSTRUCTIONS: Try to find the best complexity value to match the test data

# show the plots
plt.tight_layout()
plt.show()
