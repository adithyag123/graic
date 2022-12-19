from threading import Thread, Event
from time import sleep

event = Event()


class TestClass(object):
    def __init__(self):
        self.var = [1,2,3,4,5]

    def modify_variable(self):
        t = []
        for i in range(len(self.var)):
            t.append(self.var[i]+1)
        self.var = t
    def run(self):
        t = Thread(target=self.modify_variable)
        t.start()
        print(self.var)

class_t = TestClass()
while True:
    class_t.run()
    sleep(1)