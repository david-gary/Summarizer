import random

# create a class for binary tree


class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def add_node(self, value):
        if self.key == value:
            return False
        elif self.leftChild and value < self.key:
            return self.leftChild.add_node(value)
        elif self.rightChild and value > self.key:
            return self.rightChild.add_node(value)
        else:
            if value < self.key:
                self.insertLeft(value)
            else:
                self.insertRight(value)
            return True

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self, obj):
        self.key = obj

    def getRootVal(self):
        return self.key

    def preorder(self):
        print(self.key)
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()

    def inorder(self):
        if self.leftChild:
            self.leftChild.inorder()
        print(self.key)
        if self.rightChild:
            self.rightChild.inorder()

    def postorder(self):
        if self.leftChild:
            self.leftChild.postorder()
        if self.rightChild:
            self.rightChild.postorder()
        print(self.key)

    def search(self, value):
        if self.key == value:
            return True
        elif self.leftChild and value < self.key:
            return self.leftChild.search(value)
        elif self.rightChild and value > self.key:
            return self.rightChild.search(value)
        else:
            return False

    def printTree(self):
        print(self.key)
        if self.leftChild:
            self.leftChild.printTree()
        if self.rightChild:
            self.rightChild.printTree()

    def getHeight(self):
        if self.leftChild and self.rightChild:
            return 1 + max(self.leftChild.getHeight(), self.rightChild.getHeight())
        elif self.leftChild:
            return 1 + self.leftChild.getHeight()
        elif self.rightChild:
            return 1 + self.rightChild.getHeight()
        else:
            return 0

    def getSize(self):
        if self.leftChild and self.rightChild:
            return 1 + self.leftChild.getSize() + self.rightChild.getSize()
        elif self.leftChild:
            return 1 + self.leftChild.getSize()
        elif self.rightChild:
            return 1 + self.rightChild.getSize()
        else:
            return 1


def test_order_traversals():
    given_list = [2, 7, 5, 3, 6, 9, 4, 11, 8]
    # Create a binary tree from the random list
    bt = BinaryTree(given_list[0])
    for i in range(1, len(given_list)):
        bt.add_node(given_list[i])
    # Print the tree in preorder
    print("Preorder:")
    bt.preorder()
    # Print the tree in inorder
    print("Inorder:")
    bt.inorder()
    # Print the tree in postorder
    print("Postorder:")
    bt.postorder()


test_order_traversals()
# pre order: 10,6,2,8,7,9,14,18,16
# in order 2,6,7,8,9,10,14,16,18
