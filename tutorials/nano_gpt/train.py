# based on the tutorial from Andrew Karpathy
# source: https://www.youtube.com/watch?v=kCc8FmEb1nY


raw_dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

input_file = "input.txt"


with open(input_file, 'r') as reader:
    data = reader.read()


print(f'Loaded {len(data)} strings into the buffer')
