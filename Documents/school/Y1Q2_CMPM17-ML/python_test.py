import random

fib = [1, 1]
for i in range(10):
    fib.append(fib[-1] + fib[-2])
print(fib)


def min_and_max(lst):
    _min = float("inf")
    _max = float("-inf")
    for i in lst:
        if i < _min:
            _min = i
        if i > _max:
            _max = i
    return _min, _max


lists = [
    [random.randint(-1000, 1000) for _ in range(random.randint(5, 20))]
    for _ in range(10)
]

for l in lists:
    print(min_and_max(l))
