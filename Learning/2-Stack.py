'''Problem: Valid Parentheses
Description:

Given a string s containing just the characters '(', ')', '{', '}', '[', and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Examples:

Input: s = "()"
Output: True

Input: s = "()[]{}"
Output: True

Input: s = "(]"
Output: False

Constraints:

The input string s will only contain characters '(', ')', '{', '}', '[' and ']'.
1 <= len(s) <= 10^4
Approach:
To solve this problem, we can use a stack to keep track of opening brackets. The idea is:

Traverse the string character by character.
If an opening bracket is encountered ((, {, [), push it onto the stack.
If a closing bracket is encountered (), }, ]), check if the stack is not empty and if the top of the stack is the matching opening bracket. 
If so, pop the stack.
If the stack is empty or the top of the stack does not match the current closing bracket, the string is not valid.
After processing all characters, if the stack is empty, the string is valid. Otherwise, it is not.
'''

# valid_parentheses.py

def is_valid(s):
    # Stack to keep track of opening brackets
    stack = []
    # Dictionary to map closing brackets to opening brackets
    bracket_map = {')': '(', '}': '{', ']': '['}
    
    # Traverse each character in the input string
    for char in s:
        # If the character is a closing bracket
        if char in bracket_map:
            # Pop the top element from the stack if it's not empty, otherwise use a dummy value
            top_element = stack.pop() if stack else '#'
            
            # Check if the popped element is the matching opening bracket
            if bracket_map[char] != top_element:
                return False
        else:
            # If it's an opening bracket, push it onto the stack
            stack.append(char)
    
    # If the stack is empty, all brackets are closed properly
    return not stack

# Test the function
if __name__ == "__main__":
    test_cases = ["()", "()[]{}", "(]", "([)]", "{[]}", "["]
    for case in test_cases:
        result = is_valid(case)
        print(f"Is the string '{case}' valid? {result}")

