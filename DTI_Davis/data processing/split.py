########Transformer模型里面需要检查空格########
input_file = 'F:/D92M_test.txt'
output_file = 'F:/D92M_test_processed.txt'

def process_line(line):
    parts = line.strip().split()
    if len(parts) > 3:
        smiles = parts[0]
        sequence = ' '.join(parts[1:-1])
        interaction = parts[-1]
        return f"{smiles} {sequence} {interaction}"
    elif len(parts) < 3:
        print(f"Skipping malformed line: {line.strip()}")
        return None
    else:
        return ' '.join(parts)

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        processed_line = process_line(line)
        if processed_line:
            outfile.write(processed_line + '\n')

print(f"Processed data written to {output_file}")
