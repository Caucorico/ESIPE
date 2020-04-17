# FEUILLE DE TRICHE - Les objets

Il est possible de créer des objets en python.

## Objet simple :

Exemple :
```python
class Point:

  # Constructeur
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def display(self):
    print("Point( "+ self.x + ", " + self.y + ")")
```

La méthode __\_\_init____ définie dans l'objet est le constructeur de l'objet.
Contrairement à d'autres langages, il n'est pas nécessaire de déclarer les champs dans la classe.
Il suffit de les affecter dans une méthode.

### Créer une instance de cette objet :

Pour créer un instance, il suffit de mette le nom de la classe suivit des parenthèses et, optionnellement, des paramètres à passer au constructeur :
```python
point = Point(10, 4)
```

### Appeler une méthode de cette objet :

Appeler une méthode sur un objet se fait comme sur les autres langages.
Exemple :
```python
point_b = Point(2, 1)
# Appeler la méthode display du point :
point_b.display()
```

### Particularité des méthodes de l'objet :

La variable __self__ doit être en paramètre de chaque méthode d'instance. Cette variable __self__ représente le mot clef __this__ dans d'autres langages.
Les objets en python ressemble donc à ça :
```python
class A:

  # Constructeur
  def __init__(self):
    self.i = 0

  # Méthode a :
  def a(self):
    print(self.i)
    self.i += 1

  # Méthode b :
  def b(self, x):
    print(x)
```

### Les méthodes static en python :

Le python ne possède pas de mot clef pour indiquer qu'une méthode est statique. Il faut utiliser l'annotation __@staticmethod__.

Exemple : Nous voulons que la méthode "miaou" soit statique :

```python
class B:
  @staticmethod
  def miaou():
    print("Miaou !")
```
___
## Liste des méthodes spéciales d'objet python :
### La méthode __\_\_eq____ :

La méthode __\_\_eq____ en python correspond à la méthode equals dans d'autres langages.
Concrètement, c'est cette méthode qui permet de vérifier si deux objet sont égaux entre eux.
Redéfinir cette méthode est nécessaire lorsque l'on ne souhaite pas comparer les instance du même objet mais les valeurs des champs.
> Par défaut, python compare les instance de deux objets et renvoie __true__ s'il s'agit du même objet.

La méthode __\_\_eq____ prend en argument :
 - __self__ : Lobjet courant
 - __other__ : L'objet a comparer.


Exemple :
```python
class Point2:
  # Constructeur
  def __init__(self, x, y);
    self.x = x
    self.y = y

  # Méthode d'agalité :
  def __eq__(self, other):
    if self.x == other.x and self.y == other.y:
      return True
    else
      return False
```

### La méthode __\_\_ne____ :
En python, il existe également la méthode __\_\_ne____ qui est l’abréviation de : "not equals".
Cette méthode n'est plus nécessaire à partir de la version 3, elle est créée implicitement avec la méthode __\_\_eq____.
Cependant, il peut être nécessaire de la coder pour des versions de python inférieures ou pour des raisons plus complexe.

Elle prend les mêmes paramètres que __\_\_eq__ :
- __self__ : Lobjet courant
- __other__ : L'objet a comparer.


Exemple :
```python
class Point3:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __eq__(self, other):
    if self.x == other.x and self.y == other.y:
      return True
    else
      return False

  # Do the opposite of __eq__
  def __ne__(self, other):
    if self.x == other.x and self.y == other.y:
      return False
    else
      return True
```

### La méthode __\_\_hash____ :
Python utilise parfois le hash des objets pour effectuer ses opérations. C'est le cas de l'objet __set__ par exemple.
Le hash est calculé via la méthode __\_\_hash____.
La methode ne prend en paramètre que __self__ :
```python
class Point4:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  # La fonction calculant le hash :
  def __hash__(self):
    hash = 0
    hash += self.x * 10**0
    hash += self.y * 10**1
    return hash
```
