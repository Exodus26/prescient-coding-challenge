'''
Problem: Reverse a Linked List
Description:

Given the head of a singly linked list, reverse the list, and return the reversed list's head.

Examples:

Input: head = [1, 2, 3, 4, 5]
Output: [5, 4, 3, 2, 1]

Input: head = [1, 2]
Output: [2, 1]

Input: head = []
Output: []

Approach:
To reverse a singly linked list, you can use an iterative approach with three pointers: prev, current, and next. 
The idea is to reverse the direction of the pointers one node at a time.

Initialize: Start with prev as None and current pointing to the head of the list.
Iterate through the list: For each node, store the next node, 
reverse the current node's pointer to point to prev, then move prev and current one step forward.
Update Head: After the loop, prev will point to the new head of the reversed list.
'''

# reverse_linked_list.py

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    prev = None
    current = head
    
    while current:
        # Store the next node
        next_node = current.next
        # Reverse the current node's pointer
        current.next = prev
        # Move pointers one position ahead
        prev = current
        current = next_node
    
    # Return the new head of the reversed list
    return prev

# Helper function to create a linked list from a list
def create_linked_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

# Helper function to print linked list
def print_linked_list(head):
    current = head
    result = []
    while current:
        result.append(current.val)
        current = current.next
    print(result)

# Test the function
if __name__ == "__main__":
    test_cases = [
        [1, 2, 3, 4, 5],
        [1, 2],
        [],
        [1],
        [2, 4, 6, 8, 10]
    ]
    
    for values in test_cases:
        head = create_linked_list(values)
        print("Original list:", end=" ")
        print_linked_list(head)
        reversed_head = reverse_linked_list(head)
        print("Reversed list:", end=" ")
        print_linked_list(reversed_head)
        print("-" * 40)
