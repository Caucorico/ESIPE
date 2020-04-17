# FEUILLE DE TRICHE - Les opérateurs

## Les opérateurs en python

| Opérateur | Définition                   |
|-----------|------------------------------|
| +         | Somme                        |
| -         | Différence                   |
| %         | Modulo                       |
| /         | Division (float)             |
| //        | Division entière             |
| *         | Multiplication               |
| **        | Exponentiation               |

Exemple :

```python
x = 1 + 1
print(x) # => 2

x = 8 - 1
print(x) # => 7

x = 10 * 2
print(x) # => 20

x = 10 / 2
print(x) # => 5.0

x = 10 // 2
print(x) # => 5

x = 5 // 3
print(x) # => 1

x = 7 % 3
print(x) # => 1

x = 2**4
print(x) # => 16
```


> Quand on utilise un float dans une opération, le résultat est un float :

```python
x = 2 * 3.0
print(x) # => 6.0
```

> La priorité de calcul peut être gérée par les parenthèses :

```python
x = ( 2 + 4 ) * 3
print(x) # => 18
```
