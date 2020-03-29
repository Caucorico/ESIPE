import sys
from dist import plus_proche1
import pickle

# Réponses à la question 7 :
# a : Utiliser un fichier qui enregistre chaque résultat dans un fichier accélère l'execution du programme.
#     En effet, conserver les résultat dans le temps permet au programme de n'éxécuter la recherche qu'une seule fois.
#     Le problème, c'est si on fait beaucoup de recherches, le fichier sera volumineux. A voir combien d'espace on est prêt
#     à sacrifier. Si pas beaucoup, on pourrait ajouter des "timeouts"...
#
# b : Le problème de ce système est qu'un mot proche ortographiquement n'est pas le plus proche en terme de sens.
#     L'utilisateur peut donc être perdu.
#     Je ne sais pas comment régler ce problème...


# The memoization table
memoization_table = {}


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
    global memoization_table
    approximations = []

    for request in requests:
        if request in memoization_table:
            result = memoization_table[request]
        else:
            result = plus_proche1(request, lines)
            memoization_table[request] = result

        approximations.append((request, result))

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


def open_dictionary(filename: str):
    """
    This function inflate the dictionary if the file exists.
    :param str filename: The name of the file that contains the dictionary.
    """

    global memoization_table

    try:
        with open(filename, "rb") as file:
            try:
                memoization_table = pickle.load(file)
            except pickle.UnpicklingError:
                print("The file is not in the good format")

    except FileNotFoundError:
        open(filename, "wb+").close()


def save_dictionary(filename: str):
    """
    This function save the dictionary in the file. It create the file if it is not already exists.
    :param str filename: The name of the file that will contains the directory.
    """

    try:
        file = open(filename, "wb")
    except FileNotFoundError:
        file = open(filename, "wb+")

    pickle.dump(memoization_table, file)


def main() -> None:
    """
        This function displays all the approximations for the command line parameters.
        :rtype: None
    """

    open_dictionary("values.data")

    approximations = get_approximation_list_from_file("series_2000-2019.txt", sys.argv)
    display_terminal_approximation_list(approximations)

    save_dictionary("values.data")


main()
