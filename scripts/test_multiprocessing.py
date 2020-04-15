
#from multiprocessing.pool import ThreadPool as Pool

import sys
print(sys.executable)
def f(x):
    return x*x
print('A')
print(__name__)
if __name__ == '__main__':
    import time
    import os
    from multiprocessing import Pool as Pool
    
    print('Spawning pool')
    pool = Pool(processes=4)              # start 4 worker processes
    print('Done Spawning')
    # print "[0, 1, 4,..., 81]"
    print(pool.map(f, range(10)))

    # print same numbers in arbitrary order
    for i in pool.imap_unordered(f, range(10)):
        print(i)

    # evaluate "f(20)" asynchronously
    res = pool.apply_async(f, (20,))      # runs in *only* one process
    print( res.get(timeout=1))              # prints "400")

    # evaluate "os.getpid()" asynchronously
    res = pool.apply_async(os.getpid, ()) # runs in *only* one process
    print(res.get(timeout=1) )             # prints the PID of that process

    # launching multiple evaluations asynchronously *may* use more processes
    multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    print( [res.get(timeout=1) for res in multiple_results])
print("B")