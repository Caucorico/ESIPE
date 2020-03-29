from graph import *
import math

fibonacci5_memory = [0, 1]


def fibonacci1(n):
    if n <= 1:
        return n

    return fibonacci1(n - 1) + fibonacci1(n - 2)


def fibonacci2(n):
    if n <= 1:
        return n, 0

    (sum, sum_nbr) = fibonacci2(n - 1)
    (sum2, sum_nbr2) = fibonacci2(n - 2)

    return sum + sum2, sum_nbr + sum_nbr2 + 1


def fibonacci3(n):
    if n <= 1:
        return 0, n

    (a, b) = fibonacci3(n - 1)
    return b, a + b


def fibonacci4(n):
    if n <= 1:
        return 0, n, 0

    (a, b, sum_nbr) = fibonacci4(n - 1)
    return b, a + b, sum_nbr + 1


def fibonacci5(n):
    if n <= 1:
        return n, 0
    elif n <= (len(fibonacci5_memory) - 1):
        return fibonacci5_memory[n], 0

    res1, res1_nbr = fibonacci5(n-1)
    res2, res2_nbr = fibonacci5(n-2)
    res = res1 + res2

    fibonacci5_memory.append(res)
    return res, res1_nbr + res2_nbr + 1


def main():
    # for i in range(0, 100, 10):
    #    print(fibonacci1(i))
    #    print(fibonacci2(i))

    liste1 = [(n * 20, (fibonacci2(n)[1]) / 50) for n in range(25)]
    liste2 = [(n * 20, (n ** 2) / 50) for n in range(25)]
    liste3 = [(n * 20, (math.exp(n)) / 50) for n in range(25)]
    liste4 = [(n * 20, (fibonacci4(n)[2]) / 50) for n in range(25)]
    liste5 = [(n * 20, fibonacci5(n)[1] / 50) for n in range(25)]

    graph = Graph(500, 500, 50, 450, [liste1, liste2, liste3, liste4, liste5], ['blue', 'green', 'red', 'yellow', 'orange'])
    graph.draw_graph()


main()
