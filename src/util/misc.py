import timeit


def f_timer(function, *args):
    start = timeit.default_timer()
    results = function(*args)
    total_time = timeit.default_timer() - start
    return results, total_time
