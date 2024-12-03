import matplotlib.pyplot as plt
import numpy as np

# ======================================= LINE CHART =======================================

# Data for the graph
step_numbers = [1, 2, 3, 4, 5, 6]
without_poisoning = [1.00, 0.64, 0.59, 0.47, 0.71, 0.58]
with_poisoning = [1.00, 0.40, 0.35, 0.19, 0.36, 0.45]
with_security = [1.00, 0.74, 0.70, 0.62, 0.54, 0.57]

# Create the line graph
plt.figure(figsize=(10, 6))
plt.plot(step_numbers, without_poisoning, marker='o', label='Without Poisoning', color='green')
plt.plot(step_numbers, with_poisoning, marker='o', label='With Poisoning', color='red')
plt.plot(step_numbers, with_security, marker='o', label='With Security Measure', color='blue')

# Add labels, title, and legend
plt.xlabel('Instruction Step #', fontsize=12)
plt.ylabel('Similarity Score', fontsize=12)
plt.title('Instruction Step vs. Similarity Score', fontsize=14)
plt.legend(loc='lower left', fontsize=10)
plt.xticks(step_numbers)
plt.ylim(0, 1.1)  # Set y-axis limits to show full range

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the graph
plt.savefig("similarity_line_chart.png")
plt.show()

# ==========================================================================================


# ====================================== BAR CHART =========================================

# Data for the averages
categories = ['Without Poisoning', 'With Poisoning', 'With Security']
averages = [
    np.mean([1.00, 0.64, 0.59, 0.47, 0.71, 0.58]),
    np.mean([1.00, 0.40, 0.35, 0.19, 0.36, 0.45]),
    np.mean([1.00, 0.74, 0.70, 0.62, 0.54, 0.57])
]

# Create the bar graph
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, averages, color=['green', 'red', 'blue'], edgecolor='black')

# Add labels and title
plt.ylabel('Average Similarity Score', fontsize=12)
plt.title('Average Similarity Score Across Scenarios', fontsize=14)
plt.ylim(0, 1.1)  # Set y-axis limits to show full range

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the graph
plt.savefig("similarity_bar_chart.png")
plt.show()

# ==========================================================================================