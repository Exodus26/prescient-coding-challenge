'''
Problem: Maximum Sum Subarray of Size K
Description:

Given an array of integers nums and an integer k, find the maximum sum of any contiguous subarray of size k.

Examples:

Input: nums = [2, 1, 5, 1, 3, 2], k = 3
Output: 9
Explanation: The subarray [5, 1, 3] has the maximum sum of 9.

Input: nums = [2, 3, 4, 1, 5], k = 2
Output: 7
Explanation: The subarray [3, 4] has the maximum sum of 7.

Approach:
The sliding window technique can be applied here to find the maximum sum of a contiguous subarray of size k:

Initialize the first window sum: Calculate the sum of the first k elements. 
This will be our initial window.
Slide the window: Move the window by one element at a time. 
Add the next element in the array to the window and remove the first element of the previous window.
Update the maximum sum: Keep track of the maximum sum encountered during the process.
Return the maximum sum after sliding through the array.
'''
# maximum_sum_subarray.py

def max_sum_subarray(nums, k):
    # Edge case: if array length is less than k, return 0 (not enough elements)
    if len(nums) < k:
        return 0

    # Calculate the sum of the first 'k' elements
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide the window from start to end of the array
    for i in range(len(nums) - k):
        # Subtract the element going out of the window and add the new element coming into the window
        window_sum = window_sum - nums[i] + nums[i + k]
        # Update the maximum sum encountered
        max_sum = max(max_sum, window_sum)

    return max_sum

# Test the function
if __name__ == "__main__":
    test_cases = [
        ([2, 1, 5, 1, 3, 2], 3),
        ([2, 3, 4, 1, 5], 2),
        ([1, 1, 1, 1, 1, 1], 2),
        ([5, -1, 5, -1, 5], 2),
        ([], 3),
        ([10, 20, 30], 4)  # Edge case: k is greater than the array length
    ]

    for nums, k in test_cases:
        result = max_sum_subarray(nums, k)
        print(f"Array: {nums}, k: {k}, Maximum Sum Subarray of Size {k}: {result}")
