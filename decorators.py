def flip_the_lists(func):
    def wrapper_sort(*args):
        print("This decorator flips the lists before calling the function")
        for x in args:
            x.reverse()
        func(*args)
    return wrapper_sort
