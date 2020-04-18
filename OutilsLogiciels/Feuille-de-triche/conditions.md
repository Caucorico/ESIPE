# FEUILLE DE TRICHE - Les conditions

___

En python, il est bien entendu possible de faire des conditions.

## Les booléens :
En python, les booléens commencent par une majuscule :
 - True
 - False

Il peuvent être affecter à une variable :
```python
a = True
b = False
```

### Les opérateurs logiques :
Il est possible d'appliquer des opérateurs logiques sur les valeurs booléennes.

| Opérateur | Définition                                                           |
|-----------|----------------------------------------------------------------------|
| and       | Renvoie `True` si les deux comparaisons valent `True`                |
| or        | Renvoie `True` si une des comparaisons vaut `True`                   |
| not       | Renvoie `True` si la comparaison vaut `False` (et inversement)       |

Exemple :
```python
a = True
b = True
c = False
d = False

x = not a
print(x) # => False

x = not c
print(x) # => True

x = a and b
print(x) # => True

x = a and c
print(x) # => False

x = a or b
print(x) # => True

x = a or c
print(x) # => True

x = c or d
print(x) # => False
```

### Les opérateurs de comparaison :
Il est possible d'obtenir d'obtenir des valeurs booléennes en utilisant des opérateurs de comparaison.

| Opérateur | Définition                   |
|-----------|------------------------------|
| ==        | égalité                      |
| !=        | différence                   |
| <         | inférieur à                  |
| >         | supérieur à                  |
| <=        | inférieur ou égal            |
| >=        | supérieur ou égal            |
| is        | égalité de pointeur          |

Exemple :
```python
a = 5
b = 7
c = 5

# Egalité
x = ( a == c )
print(x) # => True

x = ( a == b )
print(x) # => False

# Différence
x = ( a != b )
print(x) # => True

x = ( a != c )
print(x) # => False

# Comparaison stricte
x = ( a < b )
print(x) # => True

x = ( a > b )
print(x) # => False

x = ( a < c )
print(x) # => False

# Comparaison
x = ( a <= b )
print(x) # => True

x = ( a >= b )
print(x) # => False

x = ( a <= c )
print(x) # => True

# Egalité de pointeur :
c = [ 1, 2 ]
d = [ 1, 2 ]

x = ( c == d )
print(x) # => True

x = ( c is d )
print(x) # => False

e = d
x = ( e is d )
print(x) # => True
```


## Les blocs if/then/else :

Il est possible d'utiliser les valeurs booléennes pour exécuter du code conditionnellement :

### Le __if__ :

Le code à l'intérieur du bloc __if__ est exécuté seulement si la condition est vérifiée.

```python
a = True
if a:
  print(" Coucou ") # Coucou displayed

a = False
if a:
  print( " Coucou " ) # Coucou not displayed
```

### Le __else__ :
Le code à l'intérieur du bloc __else__ est exécutée seulement si la condition du __if__ n'est pas respectée.

```python
a = True

if a:
  print(" Vrai ") # Vrai displayed
else:
  print(" Faux ") # Faux not displayed

a = False

if a:
  print(" Vrai ") # Vrai not displayed
else:
  print(" Faux ") # Faux displayed
```

### Le __elif__ :

Le __elif__ est une alternative au __if__. Il est exécuté si la condition du __if__ n'est pas respectée et si la condition du __elif__ est respectée.

```python
a = False
b = True

if a:
  print(" a ") # a not displayed
elif b:
  print(" b ") # b displayed
else:
  print(" c? ") # c? not displayed
```

Dès qu'une condition est acceptée, les autre ne le sont pas.

```python
a = True
b = True

if a:
  print(" a ") # a displayed
elif b:
  print(" b ") # b not displayed ( a is true ! )
else:
  print(" c? ") # c? not displayed ( a is true ! )
```
