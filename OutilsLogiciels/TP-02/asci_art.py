import sys


class ascii_letter:
    def __init__(self, letter: str, ascii: list):
        self.letter = letter
        self.ascii = ascii

        self.max_size = 0

        for line in ascii:
            if self.max_size < len(line):
                self.max_size = len(line)

        for i in range(len(self.ascii)):
            for _ in range(self.max_size - len(self.ascii[i])):
                self.ascii[i] += " "

    def get_line(self, i):
        return self.ascii[i]

    def get_size(self):
        return self.max_size

    def get_char(self):
        return self.letter


def make_char_dictionary():
    try:
        alphabet_file = open("alphabet.txt", 'r')
        symbols_file = open("symbols.txt", 'r')
        number_file = open("chiffres.txt", 'r')
    except FileNotFoundError:
        print("One of the files was not found !")
        exit(1)

    dictionary = {}

    for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
        dictionary[i] = ascii_letter(i, [alphabet_file.readline().replace('\n', '') for _ in range(8)])

    for i in ",;:!?./\"'(-)[|] ":
        dictionary[i] = ascii_letter(i, [symbols_file.readline().replace('\n', '') for _ in range(8)])

    for i in range(10):
        dictionary[str(i)] = ascii_letter(str(i), [number_file.readline().replace('\n', '') for _ in range(8)])

    return dictionary


def display_single_line_ascii_array(array: list):
    for i in range(8):
        substr = ""
        for letter in array:
            substr += letter.get_line(i)

        print(substr)


def display_line_ascii_array(array: list, line_size=80):
    last = 0
    cpt = 0

    for letter in enumerate(array):
        cpt += letter[1].get_size()

        if cpt >= line_size:
            display_single_line_ascii_array(array[last:letter[0]])
            cpt = letter[1].get_size()
            last = letter[0]


def next_space(array: list):
    cpt = 0

    if array[0].get_char() == ' ':
        return 0, array[0].get_size()

    for letter in enumerate(array):

        if letter[1].get_char() == ' ':
            return letter[0], cpt

        cpt += letter[1].get_size()

    return len(array)-1, cpt


def display_word_line_ascii_array(array: list, line_size=80):
    last = 0
    cpt = 0

    for letter in enumerate(array):
        (pos, space) = next_space(array[letter[0]:])

        if space >= line_size-cpt:
            display_single_line_ascii_array(array[last:letter[0]])
            last = letter[0]
            cpt = letter[1].get_size()

        else:
            cpt += letter[1].get_size()

    display_single_line_ascii_array(array[last:])


def make_ascii_array(dictionary: dict, string: str):
    array = []

    for letter in string:
        if letter in dictionary:
            array.append(dictionary[letter])
        else:
            array.append(dictionary["?"])

    return array


def main():
    dictionary = make_char_dictionary()

    ascii_array = make_ascii_array(dictionary, "Modifier le programme afin qu'il prenne un second argument X, qui vaut 80. La sortie devra retourner a la ligne tous les X caracteres, comme dans cet exemple. Faire attention a ne pas couper des lettres entre plusieurs lignes!")
    display_word_line_ascii_array(ascii_array)


main()
