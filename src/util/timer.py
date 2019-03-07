import timeit


def f_timer(print_function, function, *args):
    start = timeit.default_timer()
    results = function(*args)
    total_time = timeit.default_timer() - start
    print_function(f"{function.__name__} took {total_time}")
    return results
