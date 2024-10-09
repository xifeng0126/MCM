def sum_value(values):
    s = 0
    for v in values:
        s += v
    return s

array = [1,2,3,4,5]
print(sum_value(array))

def show_name(name="frank"):
    print(name)

show_name("jack")
show_name()
show_name(name="jjj")