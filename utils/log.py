# import torchvision
import shutil

import torch
import numpy as np

# import matplotlib.pyplot as plt
import time


class TimerBlock:
    """
    with TimerBlock(title) as block:
        block.log(msg)
        block.log2file(addr,msg)
    """

    def __init__(self, title):
        print(f"{title}")
        self.content = []
        self.addr = None
        self.out_path = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.time() - self.start
        units = "s"
        if duration > 60:
            duration = duration / 60.0
            units = "m"
        s = f"  [{duration:.3f}{units}] {string}"
        print(s)
        self.content.append(s + "\n")
        fid = open(self.addr, "a")
        fid.write(f"{s}\n")
        fid.close()

    def save(self, fid):
        f = open(fid, "a")
        f.writelines(self.content)
        f.close()

    def log2file(self, fid, string):
        fid = open(fid, "a")
        fid.write(f"{string}\n")
        fid.close()

    def copy2out(self, f, path=None):
        if path:
            shutil.copy2(f, path)
        else:
            shutil.copy2(f, self.out_path)


class IteratorTimer:
    """
    An iterator to produce duration. self.last_duration
    """

    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = self.iterator.__next__()
        self.last_duration = time.time() - start
        return n

    next = __next__


if __name__ == "__main__":
    with TimerBlock("Test") as block:
        block.log("1")
        block.log("2")
        block.save("../train_val_test/runs/test.txt")
