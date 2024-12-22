import time
from memory_profiler import profile, memory_usage
import matplotlib.pyplot as plt
class BNodeStar:
    """
    A node in the B* tree data structure.
    Each node maintains keys, children references, and sibling pointers.
    
    Attributes:
        keys (list): List of keys stored in the node
        isLeaf (bool): True if node is a leaf node, False otherwise
        children (list): List of child nodes
        left (BNodeStar): Reference to left sibling node
        right (BNodeStar): Reference to right sibling node
        parent (BNodeStar): Reference to parent node
    """

    def __init__(self, isLeaf=False):
        # Initialize a B* Tree node, passing in the argument whether it is a leaf node. By default, it is not.
        self.keys = []
        self.isLeaf = isLeaf
        self.children = [] 
        self.left = None
        self.right = None
        self.parent = None

    def add(self, node):
        # We are adding a child node and setting its parent reference.
        self.children.append(node)
        node.parent = self

    def findSpaceInSiblingNode(self, is_right_sibling):
        # checking if the sibling has more space for redistribution
        # it returns a boolean. True if there is space, False otherwise.
        sibling = self.right if is_right_sibling else self.left
        if not sibling: # if there is no sibling to redistribute with
            return False
        twoThirds = (2 * len(self.parent.children[0].keys)) // 3
        return len(sibling.keys) < twoThirds # check if sibling has space less than the threshold
    
    def moveKeysToSibling(self, keys_to_move, is_right_sibling):
        # move keys to either left or right sibling depending on space available
        # passes in a boolean, True if we are moving to the right sibling, False for the left sibling
        sibling = self.right if is_right_sibling else self.left
        if is_right_sibling:
            moved_keys = self.keys[-keys_to_move:] #Keys that will be moved from the end
            self.keys = self.keys[:-keys_to_move] #remove keys from the current node
            sibling.keys[0:0] = moved_keys #add keys to the beginning of the sibling
        else:
            moved_keys = self.keys[:keys_to_move]
            self.keys = self.keys[keys_to_move:]
            sibling.keys.extend(moved_keys) #append keys to the sibling node
            sibling.keys.sort() #ensure keys are sorted

    def moveChildren(self, count, is_right_sibling):
        # move children during redistribution
        # passes in a boolean, True if we are moving to the right sibling, False for the left sibling
        if self.isLeaf:
            return
            
        sibling = self.right if is_right_sibling else self.left
        if is_right_sibling:
            moved_children = self.children[-count:]
            self.children = self.children[:-count]
            sibling.children = moved_children + sibling.children
        else:
            moved_children = self.children[:count]
            self.children = self.children[count:]
            sibling.children += moved_children
            
        for child in moved_children:
            child.parent = sibling #update parent reference 
    
    def redistributeKeys(self):
        # Redistribute keys with sibling nodes to maintain B* tree properties.
        # Handles both left and right redistribution cases.
        if self.right and self.findSpaceInSiblingNode(True): #right sibling has space
            total_keys = len(self.keys) + len(self.right.keys)
            move_count = (total_keys // 2) - len(self.keys)
            self.moveKeysToSibling(move_count, True)
            self.moveChildren(move_count, True)
            
        elif self.left and self.findSpaceInSiblingNode(False): #left sibling has space
            total_keys = len(self.keys) + len(self.left.keys)
            move_count = (total_keys // 2) - len(self.left.keys)
            if move_count > 0:  
                self.moveKeysToSibling(move_count, False)
                self.moveChildren(move_count, False)
                parent_idx = self.parent.children.index(self.left)
                self.parent.keys[parent_idx] = self.keys[0] #update parent key

class BTreeStar:
    """
    B* tree implementation that maintains nodes at least 2/3 full.
    
    Attributes:
        degree (int): The minimum degree of the B* tree
        root (BNodeStar): Reference to the root node
    """
    def __init__(self, degree):
        # Initialize a B* Tree with a given degree
        self.degree = degree
        self.root = BNodeStar(True) # Creating a new node and assigning it leaf status by passing in True
    
    def threshold(self):
        # Calculates the two-thirds threshold for node capacity
        maxKeys = (2 * self.degree) - 1
        return (2 * maxKeys) // 3
    
    def redistributionAttempt(self, node, index):
        # we are determining whether we can redistribute keys with siblings
        # it returns a boolean. True if the redistribution is successful. False otherwise.
        can_redistribute = False

        # Try left sibling
        if index > 0 and len(node.children[index-1].keys) < self.threshold():
            node.children[index].left = node.children[index-1]
            node.children[index].redistributeKeys()
            can_redistribute = True
            
        # Try right sibling
        elif (index < len(node.children) - 1 and 
              len(node.children[index+1].keys) < self.threshold()):
            node.children[index].right = node.children[index+1]
            node.children[index].redistributeKeys()
            can_redistribute = True
            
        return can_redistribute
    def insert(self, key):
        # Insert a key into the B* Tree

        if len(self.root.keys) >= self.threshold(): #checking if the number of keys in the root is greater or equal to the threshold
            temp = BNodeStar(False) 
            temp.children.append(self.root)
            self.root.parent = temp 
            self.root = temp
            if len(self.root.children) > 1:
                    self.root.children[0].redistributeKeys()
            else:
                self.split(temp, 0, temp.children[0])
        self.insertNonFull(self.root, key)
    def split(self, parent, idx, child):
        # Splits a node when it exceeds the maximum capacity
        mid_index = len(child.keys) // 2
        new_node = BNodeStar(child.isLeaf)

        if not child.keys:
            return

        midKey = child.keys[mid_index]

        # splitting the node
        new_node.keys = child.keys[mid_index + 1:]
        child.keys = child.keys[:mid_index]

        # if the child is not a leaf node do:
        if not child.isLeaf:
            new_node.children = child.children[mid_index + 1:]
            child.children = child.children[:mid_index + 1]
            for grandchild in new_node.children:
                grandchild.parent = new_node
        
        # Update sibling pointers
        new_node.right = child.right
        if new_node.right:
            new_node.right.left = new_node
        new_node.left = child
        child.right = new_node
        
        # Update parent
        parent.keys.insert(idx, midKey)
        parent.children.insert(idx + 1, new_node)
        new_node.parent = parent

        for i in range(len(parent.children)):
            parent.children[i].parent = parent
            
        
    def insertNonFull(self, node, key):
        # Insert a key into a non-full node
        if node.isLeaf:
            pos = self.binarySearch(node.keys, key)
            node.keys.insert(pos, key)
            
            if len(node.keys) > self.threshold() and node.parent:
                node_idx = node.parent.children.index(node)
                self.redistributionAttempt(node.parent, node_idx)
        else:

            pos = self.binarySearch(node.keys, key) #find position using binary search algorithm
            if pos >= len(node.children): #decrements position by 1 if it exceeds the number of children
                pos = len(node.children) - 1
            if len(node.children[pos].keys) > self.threshold():
                if not self.redistributionAttempt(node, pos):
                    if len(node.children[pos].keys) == (2 * self.degree) - 1:
                        self.split(node, pos, node.children[pos])
                        if key > node.keys[pos]:
                            pos += 1
            self.insertNonFull(node.children[pos], key)

    
    def binarySearch(self, keys, target): 
        # Perform a binary search in order to find the insertion position for a key
        left, right = 0, len(keys)
        while left < right:
            mid = (left + right) // 2
            if mid >= len(keys):
                return mid
            if keys[mid] > target:
                right = mid
            else:
                left = mid + 1
        return left
    def print_tree(self, x=None, level=0):
        # Print the tree structure for debugging.
        if x is None:
            x = self.root
        print("Level", level, "Keys:", x.keys)
        level += 1
        for child in x.children:
            self.print_tree(child, level)

# Below, we are doing time and memory testing using memory profiler and time libraries
# After calculating everything, we plot the results in a display using matplotlib library
@profile
def test_insert_performance(btree, keys):
    start_time = time.time()
    for key in keys:
        btree.insert(key)
    end_time = time.time()
    return end_time - start_time

def visualize_performance(btree, keys):
    times = []
    memory_usages = []

    for i in range(1, len(keys) + 1):
        time_taken = test_insert_performance(btree, keys[:i])
        times.append(time_taken)
        mem_usage = memory_usage((btree.insert, (keys[i-1],)), max_usage=True)
        memory_usages.append(mem_usage)
    
    times = [time for time in times if isinstance(time, (int, float))]
    memory_usages = [mem for mem in memory_usages if isinstance(mem, (int, float))]


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(keys) + 1), times, label='Time Complexity')
    plt.xlabel('Number of Insertions')
    plt.ylabel('Time (seconds)')
    plt.title('Time Complexity of B* Tree Insertions')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(keys) + 1), memory_usages, label='Memory Usage')
    plt.xlabel('Number of Insertions')
    plt.ylabel('Memory (MiB)')
    plt.title('Memory Usage of B* Tree Insertions')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Test cases to evaluate our program and its effectiveness 
def main():
    # Small test case
    B = BTreeStar(3)
    keys_to_insert = [10, 20, 5, 6, 12, 30, 7, 17, 15, 9, 2, 27, 36, 8]
    visualize_performance(B, keys_to_insert)

    # Medium test case
    D = BTreeStar(3)
    keys_to_insert = [974, 637, 136, 150, 205, 707, 531, 448, 209, 972, 334, 799, 861, 428, 720, 832, 427, 584, 274, 67, 31, 34, 94, 277, 26, 94, 923, 383, 985, 754, 880, 236, 927, 758, 376, 822, 953, 855, 422, 781, 606, 146, 650, 883, 345, 808, 737, 397, 476, 242, 574, 213, 999, 830, 144, 30, 225, 515, 626, 146]
    visualize_performance(D, keys_to_insert)
    


if __name__ == '__main__':
    main()