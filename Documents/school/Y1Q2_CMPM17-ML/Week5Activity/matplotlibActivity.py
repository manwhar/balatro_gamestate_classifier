import matplotlib.pyplot as plt
import numpy as np

# Create some data. x = 100 values, evenly spaced from 0 to 10 (0, 0.1, 0.2, ... 10)
x = np.linspace(0, 10, 100)
# y = sin(x)
y = np.sin(x)

# INSTRUCTIONS: Plot the data with a line

plt.plot(x, y)
"""
Tips
    You can use plt.plot() to plot a line
    It takes in x values, and the y values that match them as the two default parameters
    Every time you plot something the plot will be added to an image. 
    You must call plt.show() to display the image, which is done below.
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for more detailed explanation
"""
plt.title("Line Plot")
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------

# INSTRUCTIONS: Plot the data with points

plt.scatter(x, y)

"""
Tips
    You can use plt.scatter() to plot points
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html for more detailed explanation
"""
plt.title("Scatter Plot")
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------

plt.scatter(x, y)
plt.plot(x, y)


"""
Tips
    You can use multiple calls before plt.show() to add multiple plots to the same image
    To distinguish, try using the color parameter (it takes a string, just type the name of the color)
"""
plt.title("Both Line and Scatter Plot")
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------

# Lets do a histogram
# Generate some numbers around 0
histogramData = np.random.randn(1000)

plt.hist(x, bins=5)

# YOUR CODE HERE

"""
Tips
    plt.hist() will generate a histogram
    You can use the bins parameter to change the number of bins
    A histogram just needs one array of x values
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html for more detailed explanation
"""
plt.title("Histogram")
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------

# Lets look at some scatter data with our scatter plots
# Generate some clustered data
clusteredDataGreen = np.random.randn(500, 2)
xGreen = clusteredDataGreen[:, 0]
yGreen = clusteredDataGreen[:, 1]
clustereeDataRed = np.random.randn(500, 2) + [3, 1]
xRed = clustereeDataRed[:, 0]
yRed = clustereeDataRed[:, 1]
clusteredDataBlue = np.random.randn(500, 2) + [-4, 4]
xBlue = clusteredDataBlue[:, 0]
yBlue = clusteredDataBlue[:, 1]

# INSTRUCTIONS: Scatter plot the three sets of data in different colors

plt.scatter(xGreen, yGreen, c="green")
plt.scatter(xRed, yRed, c="red")
plt.scatter(xBlue, yBlue, c="blue")


"""
Tips, 
    you can do multiple scatter plots in a row and then call plt.show() to display them all at once
    You can also use the s parameter to change the size of the points
    The color parameter to change the color of the points
"""
plt.title("Clustered Scatter Plot")
plt.show()

# ------------------------------------------------------------------------------------------------------------------------

# Lets look at all those plots in a grid
# INSTRUCTIONS: Make a 2x2 grid of plots from any of the previous data.

fig, axs = plt.subplots(2, 2)
# use fig to change stuff on the WHOLE FIGURE
# use axs to change stuff on individual
axs[0, 0].hist(histogramData, bins=5, edgecolor="black", label="hist of gram wow")
axs[0, 0].set_title("hooray")
axs[0, 1].plot(x, y, label="line graph uwu")
axs[1, 0].scatter(x, y, label="scatter it uppppp")
axs[1, 1].scatter(xGreen, yGreen, c="green", label="we got da MULTICOLOR SCATTER")
axs[1, 1].scatter(xRed, yRed, c="red")
axs[1, 1].scatter(xBlue, yBlue, c="blue")
axs[0, 0].legend()
fig.set_label("BIG LABEL WOW")

"""
Tips
    You can use the following to generate a grid of a, by b plot    
    fig, axs = plt.subplots(a, b)
    You can use "fig" to change stuff on the whole figure, and axs to change each individual graph
    For example, axs[0, 0] selects the top left graph, and you can then plot like normal:
    axs[0, 0].plot(x, y, label='sin(x)', color='red', linewidth=2, linestyle='dashed')
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html for more detailed explanation
    The plt.show still displays the plots
Try to make a 2x2 grid with the plots you made above
"""

# label the plots

axs[0, 0].set_title("hist")
axs[0, 1].set_title("line_grap")
axs[1, 0].set_title("")

# Use the following as a reference : axs[0, 0].set_title('Line Plot')

plt.tight_layout()  # this line of code makes the layout/format nice with even spacing

plt.show()
