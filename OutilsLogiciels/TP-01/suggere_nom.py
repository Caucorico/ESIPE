import sys
from dist import plus_proche1


def build_array(filename: str) -> list:
    """
    Create the array with all the lines of the given file.

    :param str filename: The name of the file.
    :return: list
    """
    lines = []
    file = open(filename)
    for line in file:
        lines.append(line.replace("\n", ''))

    return lines


def get_approximation_list(lines: list, requests: list) -> list:
    """
    This function return the approximation list built by parsing the given list.
    The list is normally the lines of the file.

    :param list lines: The lines of the file, the list of possibilities.
    :param list requests: The list of the requested content to looking for.
    :return: list
    """
    approximations = []

    for request in requests:
        approximations.append((request, plus_proche1(request, lines)))

    return approximations


def get_approximation_list_from_file(filename: str, requests: list) -> list:
    """
    This function return the approximation list built by parsing the given file.

    :param str filename: The name of the file that contains all the titles
    :param list requests: The list of argument to looking for in the list.
    :rtype: list
    """
    return get_approximation_list(build_array(filename), requests)


def display_terminal_approximation_list(approximations: list) -> None:
    """
    This function display the given approximations in the terminal.
    The given approximations needs to follow the following format :
    [
        (word1, approximation1)
        (word2, approximation2)
    ]

    :param list approximations: The list of approximations [(word, approximation),...]
    """
    for approximation in approximations:
        print(approximation[0] + " -> " + approximation[1])


def main() -> None:
    """
        This function displays all the approximations for the command line parameters.
        :rtype: None
    """
    approximations = get_approximation_list_from_file("series_2000-2019.txt", sys.argv)
    display_terminal_approximation_list(approximations)


main()
