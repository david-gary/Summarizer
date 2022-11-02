"""Describe the structure and pseudocode for an array-based implementation of an
index-based list that achieves O(1) time for insertions and removals at index 0,
as well as insertions and removals at the end of the list. Your implementation
should also provide for a constant-time get method."""

# Circular queue implementation


class CircularQueue:

    def __init__(self, capacity):
        self.capacity = capacity
        self.front = 0
        self.rear = 0
        self.size = 0
        self.queue = [None] * capacity

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def enqueue(self, data):
        if self.is_full():
            print("Queue is full")
            return
        self.queue[self.rear] = data
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
            return
        data = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return data

    def peek(self):
        if self.is_empty():
            print("Queue is empty")
            return
        return self.queue[self.front]

    def display(self):
        if self.is_empty():
            print("Queue is empty")
            return
        for i in range(self.size):
            print(self.queue[(self.front + i) % self.capacity])

    def get_size(self):
        return self.size
