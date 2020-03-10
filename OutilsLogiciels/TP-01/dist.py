def distance1(string1, string2):
    
    if len(string1) == 0 and len(string2) == 0 :
        return 0
        
    if len(string1) == 0 or len(string2) == 0 :
        return len(string1) + len(string2)
    
    if string1[0] == string2[0] :
        replace = distance1(string1[1:], string2[1:])
    else :
        replace = 1 +  distance1(string1[1:], string2[1:])
        
    delete = 1 + distance1(string1[1:], string2)
    insert = 1 + distance1(string1, string2[1:])
        
    return min(replace, delete, insert)
    
    
def distance1_debug(string1, string2):
    
    if len(string1) == 0 and len(string2) == 0 :
        return 0, 0
        
    if len(string1) == 0 or len(string2) == 0 :
        return len(string1) + len(string2), 0
    
    if string1[0] == string2[0] :
        res = distance1_debug(string1[1:], string2[1:])
        replace = res[0]
        replace_nbr = res[1]
    else :
        res = distance1_debug(string1[1:], string2[1:])
        replace = 1 +  res[0]
        replace_nbr = 1 + res[1]
        
    res = distance1_debug(string1[1:], string2)
    delete = 1 + res[0]
    delete_nbr = 1 + res[1]
    res = distance1_debug(string1, string2[1:])
    insert = 1 + res[0]
    insert_nbr = 1 + res[1]
        
    return min(replace, delete, insert), replace_nbr+delete_nbr+insert_nbr
    

# memoisation :
save_transformations = {}

def distance2(string1, string2):
    
    if len(string1) == 0 and len(string2) == 0 :
        return 0
        
    if len(string1) == 0 or len(string2) == 0 :
        return len(string1) + len(string2)
        
    if (string1, string2) in save_transformations :
        return save_transformations[(string1, string2)]
        
    
    if string1[0] == string2[0] :
        replace = distance2(string1[1:], string2[1:])
    else :
        replace = 1 +  distance2(string1[1:], string2[1:])
        
    delete = 1 + distance2(string1[1:], string2)
    insert = 1 + distance2(string1, string2[1:])
        
    min_value = min(replace, delete, insert)
    save_transformations[(string1, string2)] = min_value
    return min_value

    
def distance2_debug(string1, string2):
    
    if len(string1) == 0 and len(string2) == 0 :
        return 0, 0
        
    if len(string1) == 0 or len(string2) == 0 :
        return len(string1) + len(string2), 0
        
    if (string1, string2) in save_transformations :
        return save_transformations[(string1, string2)], 0
        
    
    if string1[0] == string2[0] :
        res = distance2_debug(string1[1:], string2[1:])
        replace = res[0]
        replace_nbr = res[1]
    else :
        res = distance2_debug(string1[1:], string2[1:])
        replace = 1 + res[0]
        replace_nbr = 1 + res[1]
        
    res = distance2_debug(string1[1:], string2)
    delete = 1 + res[0]
    delete_nbr = 1 + res[1]
    res = distance2_debug(string1, string2[1:])
    insert = 1 + res[0]
    insert_nbr = 1 + res[1]
    
    min_value = min(replace, delete, insert)
    save_transformations[(string1, string2)] = min_value
    return min_value, replace_nbr+delete_nbr+insert_nbr
    
    
#def plus_proche1(string, possibilities) :
#    values = []
#    
#    for possibility in possibilities :
#    return 0
    
    
    
print(distance1("abracadabra", "macabre"))
print(distance1_debug("abracadabra", "macabre"))
save_transformations = {}
print(distance2("abracadabra", "macabre"))
save_transformations = {}
print(distance2_debug("abracadabra", "macabre")[1])