'''
Problem: Two Sum
Description: Given an array of integers nums and an integer target, return the indices of the two numbers such that they add up to the target.

You may assume that each input would have exactly one solution, and you may not use the same element twice. 
You can return the answer in any order.

Example:

Input: nums = [2, 7, 11, 15], target = 9
Output: [0, 1]
Explanation: The numbers at indices 0 and 1 (2 + 7) sum up to 9.

Constraints:

2 <= nums.length <= 10^4
-10^9 <= nums[i] <= 10^9
-10^9 <= target <= 10^9
Only one valid answer exists.
Approach:
We'll use a hashing technique (via a Python dictionary) to store the numbers we've seen so far along with their indices. 
As we iterate through the array, we'll check if the difference between the target and the current number exists in the dictionary. 
If it does, we've found the pair, and we return their indices. This approach works efficiently in O(n) time complexity.
'''

# two_sum.py

def two_sum(nums, target):
    # Dictionary to store the numbers and their indices
    num_to_index = {}
    
    # Iterate over the array
    for index, num in enumerate(nums):
        # Calculate the required number to reach the target
        complement = target - num
        
        # If the complement exists in the dictionary, return the pair of indices
        if complement in num_to_index:
            return [num_to_index[complement], index]
        
        # Otherwise, store the current number with its index in the dictionary
        num_to_index[num] = index
    
    # Return an empty list if no solution is found (according to problem statement, this won't happen)
    return []

# Test the function
if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"The indices of the numbers that sum to {target} are: {result}")
