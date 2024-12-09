import matplotlib.pyplot as plt
import numpy as np

positions = [-1, 0, 1, 2, 3, 4]
mitigated_similarity = [1.0, 0.7518, 0.7984, 0.8145, 0.9106, 0.9772]
unmitigated_similarity = [1.0, 0.6232, 0.7157, 0.7681, 0.8877, 0.9713]

output_file = "poisoned_position_similarity_comparison_final.png"

plt.figure(figsize=(10, 6))
plt.plot(positions, mitigated_similarity, marker='o', label='Mitigated', color='blue')
plt.plot(positions, unmitigated_similarity, marker='o', label='Unmitigated', color='red')

plt.title('Comparison of Mitigated vs Unmitigated Similarity for Poisoned Positions', fontsize=14)
plt.xlabel('Poisoned Position', fontsize=12)
plt.ylabel('Average Similarity', fontsize=12)
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, label='Maximum Similarity (1.0)')
plt.legend(fontsize=12)
plt.grid(alpha=0.6)

plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot saved as {output_file}")


# Calculate the differences between mitigated and unmitigated similarities
differences = np.array(mitigated_similarity) - np.array(unmitigated_similarity)
difference_output_file = "similarity_difference_comparison_final.png"

# Create a bar graph for the differences
plt.figure(figsize=(10, 6))
plt.bar(positions, differences, color='purple', alpha=0.7)

# Add titles and labels
plt.title('Difference Between Mitigated and Unmitigated Similarities at Poisoned Positions', fontsize=14)
plt.xlabel('Poisoned Position', fontsize=12)
plt.ylabel('Difference in Similarity', fontsize=12)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)  # Reference line for zero
plt.grid(alpha=0.6)

# Save the plot to a file
plt.savefig(difference_output_file, dpi=300, bbox_inches='tight')
plt.close()