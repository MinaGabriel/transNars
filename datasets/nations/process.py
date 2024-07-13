import os

# Define file paths for entities and relations
entity_input_file = 'entity_ids.del'
entity_output_file = 'entity2id.txt'

relation_input_file = 'relation_ids.del'
relation_output_file = 'relation2id.txt'

# Define file paths for train, valid, and test datasets
train_input_file = 'train.del'
train_output_file = 'train2id.txt'

valid_input_file = 'valid.del'
valid_output_file = 'valid2id.txt'

test_input_file = 'test.del'
test_output_file = 'test2id.txt'

# Function to process and reverse columns in a file
def process_file(input_file, output_file):
    # Remove the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Read the input file and process the content
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Write the output file with the number of rows at the top
    with open(output_file, 'w') as outfile:
        outfile.write(f"{len(lines)}\n")  # Write the number of rows at the top
        for line in lines:
            # Split the line by tab to separate the columns
            columns = line.strip().split('\t')
            if len(columns) == 2:
                # Reverse the columns
                reversed_line = f"{columns[1]}\t{columns[0]}\n"
                # Write the reversed columns to the output file
                outfile.write(reversed_line)

# Function to swap the second and third columns in a file
def swap_columns(input_file, output_file):
    # Remove the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Read the input file and process the content
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Write the output file with the number of rows at the top
    with open(output_file, 'w') as outfile:
        outfile.write(f"{len(lines)}\n")  # Write the number of rows at the top
        for line in lines:
            # Split the line by tab to separate the columns
            columns = line.strip().split('\t')
            if len(columns) == 3:
                # Swap the second and third columns
                swapped_line = f"{columns[0]}\t{columns[2]}\t{columns[1]}\n"
                # Write the swapped columns to the output file
                outfile.write(swapped_line)

# Process entity_ids.del to create entity2id.txt
process_file(entity_input_file, entity_output_file)

# Process relation_ids.del to create relation2id.txt
process_file(relation_input_file, relation_output_file)

# Process train.del to create train2id.txt
swap_columns(train_input_file, train_output_file)

# Process valid.del to create valid2id.txt
swap_columns(valid_input_file, valid_output_file)

# Process test.del to create test2id.txt
swap_columns(test_input_file, test_output_file)

print(f"Processed data has been written to {entity_output_file}, {relation_output_file}, {train_output_file}, {valid_output_file}, and {test_output_file}")
