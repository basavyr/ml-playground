# based on the tutorial from Andrew Karpathy
# source: https://www.youtube.com/watch?v=kCc8FmEb1nY


raw_dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

input_file = "input.txt"


with open(input_file, 'r') as reader:
    data = reader.read()


def get_first_n_sequence(input_data: str, n: int = 100):
    """
    - Prints the first `n` characters of the input sequence in `input_data`
    """
    print(f'Loaded {len(data)} strings into the buffer')
    print(f'First {n} characters:\n{input_data[:n]}')


get_first_n_sequence(data, 1000)
