import os


def generate_dataset(files: list[str]):
    output_file = "dataset.jsonl"
    data = []
    for file in files:
        with open(file, 'r') as reader:
            data.extend(reader.readlines())

    with open(output_file, 'w+') as writer:
        writer.writelines(data)


def main():

    output_dir = "./volumes"

    files = [f'{output_dir}/{file}' for file in os.listdir(output_dir)]

    generate_dataset(files)


if __name__ == "__main__":
    main()
