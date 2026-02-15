from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

code = """

class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        # Dictionary to store value: index pairs
        prevMap = {} 
        
        for i, n in enumerate(nums):
            # Calculate the required number to reach the target
            diff = target - n
            
            # Check if the complement exists in our map
            if diff in prevMap:
                return [prevMap[diff], i]
            
            # Store the current number's index for future lookups
            prevMap[n] = i


"""

splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=100,
    chunk_overlap=0,
    language=Language.PYTHON,
)

res = splitter.split_text(code)

print(res[0])