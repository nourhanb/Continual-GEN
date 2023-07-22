import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('/ubc/ece/home/ra/grads/nourhanb/Documents/ood-detection-main/results_GMM/SimCLR/cp_results_dmf.csv')

# Define the cluster labels
clusters = ['GT', '5', '10', '15', '20', '25', '30']

# Create a list to store the data for each cluster
cluster_data = []

# Iterate over each cluster and extract the corresponding data
for cluster in clusters:
    group = data[data.iloc[:, 2] == cluster].iloc[:, [1]]
    # Skip rows with empty values in the second column
    group = group.dropna()
    if not group.empty:
        cluster_data.append(group.values.flatten())

# Customize the boxplot format
color_dict = {'GT': 'blue', '5': 'red', '10': 'green', '15': 'orange', '20': 'purple', '25': 'cyan', '30': 'magenta'}

# Plot the box plots for all clusters in a single figure
boxplot = plt.boxplot(cluster_data, labels=clusters)

for patch, cluster in zip(boxplot['boxes'], clusters):
    patch.set(color=color_dict[cluster])

# Set the line width of the boxes
line_width = 3
for box in boxplot['boxes']:
    box.set_linewidth(line_width)

plt.title('SimCLR with GMM')
#plt.xlabel('Clusters')
#plt.ylabel('P Values')
plt.grid(axis='y', linestyle='--')
# Set the y-axis limits
plt.ylim(0.1, 1.05)
# Remove the y-axis
plt.xticks(color='w')

plt.savefig('./figures_DMF/SimCLR_CP_GMM_dmf.png')
plt.show()
