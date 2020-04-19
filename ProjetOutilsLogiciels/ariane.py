from logic import Logic
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage : python3 ariane.py <map_file>")
        exit()

    logic = Logic(sys.argv[1])
    logic.start()
    logic.rounds()


main()
