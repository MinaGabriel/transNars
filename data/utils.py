import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def check_folder_and_files(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        return f"Folder '{folder_path}' does not exist."

    # List of expected files
    required_files = ['entity2id.txt', 'relation2id.txt',
                      'test2id.txt', 'train2id.txt', 'valid2id.txt']

    # Check for the presence of each file
    missing_files = [file for file in required_files if not os.path.isfile(
        os.path.join(folder_path, file))]

    if missing_files:
        print(f"Missing files in '{folder_path}': " + ", ".join(missing_files))
        return False
    else:
        print(f"All required files are present in '{folder_path}'.")
        return True
    
def write_arrays_to_file(filename, batch_h, batch_r, batch_t):
    # Ensure all arrays have the same length
    assert len(batch_h) == len(batch_r) == len(batch_t), "Arrays must have the same length"

    with open(filename, 'w') as f:
        for h, r, t in zip(batch_h, batch_r, batch_t):
            f.write(f"{h} {r} {t}\n")

# save training data 

def save_training_data(losses, normalized_mean_ranks, filename):
    df1 = pd.DataFrame({'losses': losses, 'mean_ranks': normalized_mean_ranks})
    with pd.HDFStore(f'./models/logs/{filename}.h5', mode='w') as store:
        store.put('df1', df1) 
        
        
def plot_loss_and_mean_rank(losses, mean_ranks, validation_rate, saved_on_epoch, filename):
    
    
    # Normalize mean ranks to the same length as epochs/losses
    normalized_mean_ranks = np.full(len(losses), np.nan)
    validation_steps = range(0, len(losses), validation_rate)
    
    for i, step in enumerate(validation_steps):
        normalized_mean_ranks[step] = mean_ranks[i]

    save_training_data(losses, normalized_mean_ranks, filename)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot loss on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(range(len(losses)), losses, label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot mean rank on secondary y-axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Mean Rank', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(len(losses)), normalized_mean_ranks, label='Mean Rank', color=color)
    ax2.plot(validation_steps, mean_ranks, color=color)  # Add markers for validation points
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # Add a vertical line for early stopping
    plt.axvline(x=saved_on_epoch, color='red', linestyle=':', linewidth=2, label='Early Stopping')
    plt.legend(loc='upper right')

    plt.title('Training Loss and Mean Rank over Epochs')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Save the plot as an image
    plt.savefig('./models/plot/' +filename+'.png')
    plt.show()

"""
takes a file_path and returns a dictionary of {"entity_name": entity_id} or {"relation_name": relation_id}
"""

def generate_dictionary(file_path: str) -> dict:
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Use regex to split based on any sequence of whitespace
            parts = re.split(r'\s+', line.strip())
            if len(parts) == 2:
                value, id = parts
                try:
                    id = int(id)
                    dictionary[value] = id
                except ValueError:
                    print(f"Warning: ID is not an integer in line: {line.strip()}")
            else:
                print(f"Warning: Line does not contain exactly two elements: {line.strip()}")
    return dictionary