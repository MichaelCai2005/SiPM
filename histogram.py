import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))


data = np.loadtxt("Datasets/2024.10.24 SiPM Count.csv", dtype=float, delimiter=",")
dataDark = np.loadtxt("Datasets/2024.10.24 Dark Count.csv", dtype=float, delimiter=",")


x0_values=dataDark[:,0]
y0_values=dataDark[:,1]

initial_guess = [max(y0_values), x0_values[np.argmax(y0_values)], 10]
popt, _ = curve_fit(gaussian, x0_values, y0_values, p0=initial_guess)
a_fit, mu_fit, sigma_fit = popt
x_fit = np.linspace(min(x0_values), max(x0_values), 1000)
y_fit = gaussian(x_fit, *popt)

plt.figure(figsize=(10, 6))
plt.plot(x0_values, y0_values, 'bo', label="Data")
plt.plot(x_fit, y_fit, 'r-', label=f"Gaussian Fit: $\\mu={mu_fit:.2f}$, $\\sigma={sigma_fit:.2f}$")
plt.xlabel('ADC Channels')
plt.ylabel('Counts')
plt.title('Histogram of ADC Channels with Gaussian Fit')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(x1_values, y1_values, marker='o', linestyle='-')

plt.xlabel('ADC Channels')
plt.ylabel('Counts')
plt.title('Histogram of ADC Channels')
plt.show()


x1_values= data[:,0]
y1_values = data[:,1]



plt.plot(x0_values, y0_values, marker='o', linestyle='-')
plt.xlabel('ADC Channels')
plt.ylabel('Counts')
plt.title('Dark Count Histogram of ADC Channels')
plt.show()


