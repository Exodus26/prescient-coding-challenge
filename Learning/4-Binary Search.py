'''
Problem: Find First and Last Position of Element in Sorted Array
Description:

Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value. 
If the target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.

Examples:

Input: nums = [5, 7, 7, 8, 8, 10], target = 8
Output: [3, 4]

Input: nums = [5, 7, 7, 8, 8, 10], target = 6
Output: [-1, -1]

Input: nums = [], target = 0
Output: [-1, -1]

Approach:
Since the array is sorted, we can efficiently find the first and last position of the target using binary search. 
We'll use two separate binary searches:

One to find the first occurrence of the target.
One to find the last occurrence of the target.
Both searches will run in O(log n) time complexity, and since they are independent, the overall complexity remains O(log n).'''

# find_first_and_last_position.py

def find_first_and_last(nums, target):
    # Helper function to find the first position of the target
    def find_first(nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                # Check if this is the first occurrence
                if mid == 0 or nums[mid - 1] != target:
                    return mid
                right = mid - 1
        return -1

    # Helper function to find the last position of the target
    def find_last(nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                # Check if this is the last occurrence
                if mid == len(nums) - 1 or nums[mid + 1] != target:
                    return mid
                left = mid + 1
        return -1

    # Find the first and last positions using the helper functions
    first_pos = find_first(nums, target)
    last_pos = find_last(nums, target)
    
    return [first_pos, last_pos]

# Test the function
if __name__ == "__main__":
    test_cases = [
        ([5, 7, 7, 8, 8, 10], 8),
        ([5, 7, 7, 8, 8, 10], 6),
        ([], 0),
        ([1, 2, 2, 2, 2, 3, 4, 5, 6], 2),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 10)
    ]
    
    for nums, target in test_cases:
        result = find_first_and_last(nums, target)
        print(f"Array: {nums}, Target: {target}, First and Last Positions: {result}")
