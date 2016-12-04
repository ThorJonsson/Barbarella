'''
A simple python script that takes a number
of arguments from the command line and produces
output.

Here we compute the Collatz sequence for a given
integer and report the stopping time.
see:
    https://en.wikipedia.org/wiki/Collatz_conjecture
'''
import sys

def collatz(n, max_steps=1000):
    steps = 0
    # we could do this recursively, but then we'd need
    # to modify python's recursion limit with
    # sys.setrecursionlimit(limit)
    while n != 1 and steps < max_steps:
        if n % 2 == 0:
            n = n / 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps, n

def usage():
    print "Usage: %s integer [max_steps]" % sys.argv[0]
    print "Where both integer and max_steps must be interpretable as integers."
    print "if not provided, max_steps=1000"

if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        usage()
        sys.exit(-1)

    # sys.argv is a list of strings, so we must convert them
    # to integers
    try:
        n = int(sys.argv[1])
        if len(sys.argv) == 3:
            max_steps = int(sys.argv[2])
        else:
            max_steps = 1000
    except ValueError:
        usage()
        sys.exit(-1)

    steps, final_n = collatz(n, max_steps)

    if final_n == 1:
        print "Collatz sequence for %s reached final value of 1 in %s steps." \
                % (n, steps)
    else:
        print "Collatz sequence for %s failed to reach 1 in %s steps." \
                % (n, max_steps)
