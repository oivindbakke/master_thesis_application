from decorators import flip_the_lists

list1 = ["Øivind", "Frank", "Fred"]
list2 = ["Master Yoda", "Is", "Talking"]
print(*list1)
print(*list2)


@flip_the_lists
def printlists(*args):
    for x in args:
        print(*x)

printlists(list1, list2)

