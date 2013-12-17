

def buffered_iter(arr, buffer_size=1000):
    for idx in xrange(0, len(arr), buffer_size):
        for e in arr[idx:idx+buffer_size]:
            yield e