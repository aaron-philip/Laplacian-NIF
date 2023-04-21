"""
Used to construct a summary of all the models training losses and time
by searching the output slurm files using regex.

Simply run in the directory of slurm files with 'python model_summaries.py'
"""

import os
import re

# Set the directory where the files are located
directory = "."

# Create empty lists to store the values of d1, d2, and training time
d1_list = []
d2_list = []
training_time_list = []

# Define the regular expressions to search for
d1_regex = r"Running: (\S+)"
d2_regex = r"loss: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
training_time_regex = r"Training Time: \s*(\d+(?:\.\d+)?)"

# Compile the regular expressions for efficiency
d1_pattern = re.compile(d1_regex)
d2_pattern = re.compile(d2_regex)
training_time_pattern = re.compile(training_time_regex)

# Loop through all files in the directory with names starting with "slurm-"
for filename in os.listdir(directory):
    if filename.startswith("slurm-"):
        with open(os.path.join(directory, filename), "r") as f:
            print(filename)
            file_contents = f.read()
            # Use regular expressions to search for d1, d2, and training time values
            d1_match = d1_pattern.search(file_contents)
            if d1_match:
                d1_value = d1_match.group(1)
            else:
                d1_value = "NaN"
            
            d2_match = re.findall(d2_regex, file_contents)
            if d2_match:
                d2_value = d2_match[-1]
            else:
                d2_value = "NaN"

            training_time_match = training_time_pattern.search(file_contents)
            if training_time_match:
                training_time_value = training_time_match.group(1)
                
            else:
                training_time_value = "NaN"

            # Add the values to the appropriate lists
            d1_list.append(d1_value)
            d2_list.append(d2_value)
            training_time_list.append(training_time_value)

# Combine d1, d2, and training time lists into a list of tuples
output_list = list(zip(d1_list, d2_list, training_time_list))
d2_list = [float(x) for x in d2_list]

# Find the lowest loss value and corresponding d1 and training time entries
min_d2 = min([x for x in d2_list if isinstance(x, float)])
min_d1 = d1_list[d2_list.index(min_d2)]
min_training_time = training_time_list[d2_list.index(min_d2)]

# Write the output to a file with columns for d1, d2, and training time
#with open("model_summaries.txt", "w") as f:
#    f.write("Model:" + "\t" + "Loss (MSE):" + "\t" + "Training Time (s):" + "\n")
#    for triple in output_list:
#        f.write(triple[0] + "\t" + str(triple[1]) + "\t" + triple[2] + "\n")
#    
#    f.write("\nLowest Loss: " + ("{:.5e}".format(min_d2)) + "\t Corresponding Model: " + min_d1 + "\t Training Time: " + min_training_time)
with open("model_summaries.txt", "w") as f:
    # Write column titles
    f.write("{:<20s}\t{:<12s}\t{:<20s}\n".format("Model", "Loss (MSE)", "Training Time (s)"))

    # Write data
    for triple in output_list:
        f.write("{:<15s}\t{:<15s}\t{:<20s}\n".format(triple[0], triple[1], triple[2]))
	
    # Write lowest loss value and corresponding d1 and training time entries
    f.write("\nLowest Loss: " + ("{:.5e}".format(min_d2)) + "\t Corresponding Model: " + min_d1 + "\t Training Time: " + min_training_time)