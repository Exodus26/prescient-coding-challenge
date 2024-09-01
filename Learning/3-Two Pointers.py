'''
Problem: Container With Most Water
Description:

You are given an array height of non-negative integers where each integer represents a point at coordinate (i, height[i]). 
n vertical lines are drawn such that the two endpoints of the line i are at (i, 0) and (i, height[i]).
Find two lines that together with the x-axis form a container that holds the most water.

Return the maximum amount of water a container can store.

Note: You may not slant the container, and n is at least 2.

Example:

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The vertical lines at positions 1 and 8 (0-indexed) form a container that holds the most water. 
The area of water it can contain is min(8, 7) * (8 - 1) = 49.

Approach:
We can solve this problem efficiently using the two pointers technique:

Initialize two pointers: One at the beginning (left = 0) and one at the end (right = len(height) - 1) of the array.
Calculate the area: The area is determined by the shorter line among the two pointers times the distance between them (right - left).
Move the pointer with the smaller height: To try and find a higher line that could potentially hold more water.
Update the maximum area: Keep track of the maximum area encountered during this process.
Continue until the pointers meet: When they do, we've considered all possible containers.
'''
# container_with_most_water.py

def max_area(height):
    left = 0
    right = len(height) - 1
    max_area = 0
    
    while left < right:
        # Calculate the width between the two pointers
        width = right - left
        # Calculate the area with the shorter height
        current_area = min(height[left], height[right]) * width
        # Update the maximum area if the current area is larger
        max_area = max(max_area, current_area)
        
        # Move the pointer with the smaller height to find potentially higher lines
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

# Test the function
if __name__ == "__main__":
    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    result = max_area(height)
    print(f"The maximum amount of water a container can store is: {result}")
