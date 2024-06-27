import argparse


def main():
    parser = argparse.ArgumentParser(description="Make plots from hidden layer cosine similarity measurements")
    parser.add_argument("data", type=str, help="Path to the data file")


# Load the data
if __name__ == "__main__":
    main()
