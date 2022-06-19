LIST OF ALGOS AND PATTERNS 
=================================
When to use TREE STYLE-- DIRECTED TREES 
When to use GRAPH STYLE-- UNDIRECTED TREES & GRAPHS 

Why do i need not visited in TREE while in graph I do need?
Actual trees are DIRECTIONAL so we cant go back to PARENT and its a TREE so no loops.
GRAPHS -- CAN GO BACK TO PARENT and go find Loops. EVEN if I have a TREE GRAPH (no loops but undirected) I still need visited because can go back to parent.
=================================
ALGO GRAPHS
1. ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK/GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK -- Done
2. ALGO_NAME: TREE_STYLE_DFS_HEAD_RECURSIVE(with how to convert to memo)/ 
    Note: GRAPH_STYLE_DFS_HEAD_RECURSIVE(doesnt exist) -- Done
3. ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_INORDER -- Done
4. ALGO_NAME: TREE_STYLE_BFS_ITERATIVE/ ALGO_NAME: GRAPH_STYLE_BFS_ITERATIVE -- Done
7. ALGO_NAME: GRAPH_STYLE_DFS_TAIL_RECURSIVE 
9. ALGO_NAME: DIRECTED_GRAPH_STYLE_DFS_TOPO -- Done
10.ALGO_NAME: UNION_FIND -- Done
11.ALGO_NAME: BOTTOM_UP_TABULATION_2D_BACKWARDS -- Done
---------------------------
5. ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION
6. ALGO_NAME: TREE_STYLE_TAIL_RECURSION
---------------------------
11. ALGO_NAME: BINARY_SEARCH_STANDARD -- Done
12. ALGO_NAME: SLIDING_WINDOW_VIOLATION_CORRECTION 
13. ALGO_NAME: STACK_INTERVALS_DOUBLE_WHILE_VIOLATION_CORRECTION
14. ALGO_NAME: HEAP_THEORY -- Done
15. ALGO_NAME: QUICKSELECT, QUICKSORT -- Done
15. ALGO_NAME: BASE_EXPLORER_METHOD -- Done
16. ALGO_NAME: MATRIX THEORY -- Done
===================================================================================================
---------> 1) ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK
1) logging=True
2) No root arg in dfs defn. 
3) stack=[(hardcoded initiation values)]
4) while stack
5) breakdown of stack.pop
6) neighbors create using by active use of breakdown values
7) print current & neighbors here
7) for neighbors in neighbors
8) neighbor breakdown into next values 
9) if simple dfs, keep appending if neighbor exist, other break into continuation vs ans condition for bactrack.
10) 
10) continuation condition will be on next variables. ans will also use next variables. while we append neighbors into stack.
11) Do I need to do any operation while iteration, use current, print stack& visited after visited has been marked.
12) SYNTAX CHECK!! LOGIC CHECK!!

The print statements this way will show every stack addition is going. 

---------> 1) ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE 
------------------------------------------------
1. Write TREE_STYLE_DFS_ITERATIVE_BACKTRACK
2. add visited & check visited,Before adding to stack check boundary condn,  and check stack  (3 things extra) right before adding to stack. 
4. Mark visited after pop in current location and do your other operations after pop

  
===================================================================================================

---------> 2) ALGO_NAME: TREE_STYLE_DFS_HEAD_RECURSIVE
LOGIC POINTS
a) logging=False at the top
b) BASE CASE AT TOP-- Base case defn will also use "root" or broken down values from "root"
c) FIND NEIGHBORS OF ROOT-- We actively use "root"/broken down variables from "root"  to fill up neighbors 
d) print the root, print the neighbors created
e) recurse ON NEIGHBORS and assume we magically have the answer
f) Using the neighbor results, create root result and then return that 
SYNTAX POINTS
g) "STATE" is always represented by "root" and hardcoded values for root are passed from calling function
h) Next MEMOIZE if needed
    1) add self.memo to the calling function. 
    2) Between the Base case and neighbor creation check if "root" is in self.memo and return it 
    3) Instead of returning the answer equate it to self.memo[root] and return self.memo[root]
i) Test on small cases to check working and then debug from there
j AN IMPORTANT DIFFERENCE BETWEEN TREE_STYLE_DFS_HEAD_RECURSIVE and TREE_STYLE_DFS_ITERATIVE 
IS THAT THERE IS NO CHECKING BEFORE Running dfs on the stack, we deal with in base case


class Solution(object):
    def dfsR(self, root):
        if not root:            ### Base case comes first and returns the ANSWER
            return []
        neighbors=[root.left,root.right]
        list1=[root.val]
        for neighbor in neighbors:  ### FIFO 
            a=self.dfsR(neighbor)   ### ASSUMPTION THAT WE KNOW THE ANSWER
            list1=list1+a
            
        return list1            ### RETURN TYPE SHOULD BE SAME AS BASE CASE
======================================================================
---------> 2) GRAPH_STYLE_DFS_HEAD_RECURSIVE
1. Write TREE_STYLE_HEAD_RECURSIVE
2. add visited & check visited,Before adding to stack check boundary condn,  and check stack  (3 things extra) right before adding to stack. 

======================================================================
---------> 3) ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_INORDER
1. I start with empty stack 
2. current is hardcoded above
3. while stack becomes "while current or stack"
4. if current: then stack append current & current.left, no printing 
5. else: current=stack.pop() and now we print and current=current.right
---------------------------------------------------------------------------------
class Solution(object):
    def inorderTraversal(self, root):
        ret=[] 
        current=root
        stack=[]
        while current or stack:    #while loop goes on until stack is exhausted. To initialize  we give current
            if current:                 # keep going left until possible
                stack.append(current)
                current=current.left
            else:
                current=stack.pop()    # when we cannot go left any more pop ##leaf node
                ret.append(current.val)  # store the value after popping
                current=current.right        # go right whenever left child is None
        return ret
==========================================================================================
---------> 4) ALGO_NAME: TREE_STYLE_BFS_ITERATIVE/ ALGO_NAME: GRAPH_STYLE_BFS_ITERATIVE
1. Only a couple of changes from DFS_ITERATIVE:
    a) I use queue in place of stack and insert(0,neighbor) 
    b) level=0 
    c) for i in range(len(queue))
    d) level+=1 OUTSIDE OF FOR LOOP
    e) And we remember where to return levels see_below
    f) Same 3 things extra for GRAPH BFS from TREE BFS: 

class Solution(object):     ### JUST SEE LEVELS THEORY HERE IGNORE REST
    def jump(self, nums):
        n=len(nums)
        queue=[0]
        level=0
        visited={}
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                if current==n-1:            ### SINCE WE CHECKED ON NEIGHBOR CURRENT HASNT BEEN CHECKED YET
                    return 0  
                #print(current,nums[current])
                neighbors=[current+jump_score for jump_score in range(1,nums[current]+1,1)]
                #print(neighbors)
                for neighbor in neighbors:
                    if neighbor==n-1:
                        return level  #### CHECKING LEVEL HERE WILL BE -1,0,1
                    queue.insert(0,neighbor)
                    visited[neighbor]=1
            print(level)    ### CHECKING LEVEL HERE WILL BE 0,1,2
            level+=1        ### INCREASE HERE BY DEFAULT
        print(level)    ### CHECKING LEVEL HERE WILL BE 1,2,3
        return False 
=======================================================================================================
---------> 11) ALGO_NAME:BOTTOM_UP_TABULATION_2D_BACKWARDS 
LOGIC POINTS
1. Decide you will use range or indexes to iterate on rows and columns based on recurrence reln.
   Here:  
   Range for rows is index while column is in actual column range. This is based on the recurrence relationship which      uses index for i but actual value for column. 
2. Fill the base cases first.
    Here the nth row can be filled and the 0th column can be filled. 
3. Decide what is the sequence of calculation.
    Of course we need to go backward in rows.
    Why forward in columns. Because we use left side elements to fill right side.
4. Out of bounds check: Can we go out of bounds? yes. Add a check for that and deal with it.
SYNTAX POINTS
5. When we are in dp[i][k]. No conversion is needed for i because we use index directly for i.
   But conversion is needed for k. 
6. When I refer the actuals array of rows,coins here, conversion is needed again. 



===================================================================================
Things to remember in these algos
1. DFS graph-- visited check and stack check before append, visted mark after pop 
2. BFS graph--visited check and queue check before append, visted mark after pop. 
BFS levels counting, position of level+=1
3. DFS Tree -- exists check before append,append list after pop,
4. BFS Tree -- exists check before append, append list after pop
===================================================================================================

===================================================================================================
QUICK SORT 
############################################ QUICK SORT ALGORITHM
Goal:  move all elements smaller than pivot to left of pivot,larger to right of pivot
## base is at left most, pivot is at right most , iterate from base to pivot "searching for smaller elements"
## while iteration only two things can happen, 
## less than pivot, we want things less than pivot to left of base, so we swap exp and base and move base one up
## greater than pivot, this is already good 
## once the iteration is over base is in the correct place so swap with it. 
Now goal is achieved, right side is all greater than pivot ..left side is all smaller than pivot, 
We have sorted that index correctly.Now we recurse on both sides to sort others
# Note the element which was in pivot will go to its correct position  
    P
3,1,2   ## greater than pivot, base stuck
B   
exp

    P
3,1,2   ## less than pivot 
B 
  exp
    P
1,3,2   ## swap happened so that smaller element goes to left 
    exp
  B


class Solution(object):
    def partition(self,array, base, pivot):  ## this function takes the right most element and puts it in its correct 
        for exp in range(base,pivot):   ## iterate over the from base to pivot but excludes pivot
            if array[exp] <= array[pivot]:
                array[exp], array[base] = array[base], array[exp]  ###swap with base, so that everything to right of base is bigger
                base += 1
        array[base], array[pivot] = array[pivot], array[base]
        return base                  ## returns the pivot

    def quicksort(self,array, left, right):
        if left < right:
            pivot = self.partition(array, left, right)  ## returns the index at which correct sorting has happened
            self.quicksort(array, left, pivot-1)        ## now sort in other parts
            self.quicksort(array, pivot+1, right)
    
    def sortColors(self, nums):
        self.quicksort(nums,0,len(nums)-1)        
----------------------------------------------------------------------------------------------      
Kthsmallest_quicksort
------------------------------------------------------------------------------------------   
class Solution(object):
    def partition(self,array, base, pivot):  ## this function takes the right most element and puts it in its correct 
        for exp in range(base,pivot):   ## iterate from base to pivot, exclude pivot
            if array[exp] <= array[pivot]:
                array[exp], array[base] = array[base], array[exp]  ###swap with base, so that everything to right of base is bigger
                base += 1
        array[base], array[pivot] = array[pivot], array[base]       ### this step is different than base, explorer
        return base                  ## returns the pivot

### This function gives the kth smallest value
### We can still get smallest k elements using array[:k] THESE ELEMENTS ARENT  SORTED


    def Kthsmallest_quicksort(self,array, left, right,k):           ### REMEMBER THIS FUNCTION AS A MODIFICATION TO QUICK SORT FUNCTION
        if left < right:
            pivot = self.partition(array, left, right) 
            #kth smallest number lives on k-1 index
            if pivot>k-1:                   ### In regualar quicksort we iterate on both sides, kth smallest 
                return self.Kthsmallest_quicksort(array, left, pivot-1,k)        ## now sort in other parts
                                                ## MEMORY TRICK OPPSITE SIDE HERE
            elif pivot<k-1:
                return self.Kthsmallest_quicksort(array, pivot+1, right,k)
            else:
                return array[pivot]
        else:                   ## sometimes no sorting needed due to single element remaining
            return array[left]
    ### gives the kth smallest value
   
    def findKthLargest(self, nums, k):
        return self.Kthsmallest_quicksort(nums,0,len(nums)-1,len(nums)-k+1) ## KTH LARGEST IS N-K+1 smallest
        
===================================================================================================



Linklist basics 

1. Traversing a link list moving once
node=head
while node:        ### Simple if the node exists we print it. Since it exists we can still can .next
	print(node.val)
	node=node.next

## at the end of the loop 'node' is None. use node.next if you want to stop one before  ## Remember

2. Traversing a LL moving twice 
node=head
while node and node.next:     #### We first check for NODE if it exists. If node exists, then .next is valid (can be None but still valid ) so  we check for NODE.NEXT
                              #### it that also exists we can call node.next.next  
    print(node.val)
    node=node.next.next

## At the end of this loop node is either at NULL(even) or at last Node (odd) ## Remember

LINK LIST RULE!!!!
If something exists then its .next can be called. so if you are calling .next of something you have to check before .next exists or not 
===================================================================================================

Matrix traversal and swapping 
--------------------------------------         
Creation:
mat2=[[0 for x in range(c)] for y in range(r)]
Other ways will give ERROR

Iteration : 
# In a symmetric matrix, you can traverse only till the diagonal by :
for i in range(m):
    for j in range(i,n,1):       #### upper right triangle including diagonal   


for i in range(m):
    for j in range(i):           ##### lower left triangle excluding diagonal

for i in range(m/2):             ##### Vertical swap 
    for j in range(n):
      matrix[i][j],matrix[m-1-i][j]= matrix[m-1-i][j],matrix[i][j]

for i in range(m):               #### horizontal swap 
    for j in range(n/2):              
        A[i][j],A[i][n-1-j]=A[i][n-1-j],A[i][j]


### In a non symmetric matrix diagonal doesnt make much sense 

Swapping across diag1 i,j becomes j,i  == diag1 in the  +ve slope diag (the one which looks like -ve slope is +ve)
Swapping across diag2 i,j becomes n-1-j,n-1-i ##checked
-----------------------------------------------------------
We can use cordinate geometry to get relationship on a line on the matrix, get two points i,j 
m=i2-i1/j2-j1 and put one point to get c 
Once we have the equation, how to generate points along the line ?
loop on one variable range(m)/range(n) and generate the other variable, check if this variable is within limits and if it is then we have our number




Commonly used Code Structures
--------------------------------------------------------
1. I want to do something only if the loop completes

### Check if 5 is in a list 
for i in [1,2,5,3]:
	if i==5:
		return True
return False

### What if I cannot use return ?? Only append the list if it doesnt have 5  
res=[]
list1
for i in list1:
    if i==5:
        break
else:
    res.append(list1)

######## NOTE HOW I CAN USE A FOR LOOP WITH AN ELSE CLAUSE !!!! THIS IS SOMETHING NEW !!!!
######## Here the else wil run ONLY AFTER the LOOP COMPLETES IF IT BREAKS IT SKIPS!!! 

This could also have been achieved by using a boolean which turns to false if something happens
--------------------------------------------------------

Conditions and structures in Python 

1. if else structures
-------------------
1. 
if cond1:
    if cond2:
      if cond3:
This is equivalent of chaining all with and.  (if cond1 and cond2 and cond3)

"or" N "and" Python
=======================
Sequence of cond is IMPORTANT: and OR statement. It is possible that one condition gives rise to an error. if that is the case 
then it should be SECOND not first. 

OR statement (x or y)
=====================================
y can have error if x is True !!!!

AND statement(x and y)
================================
 y can have error if x is False !!!


Deciding between "or" and "And" (V IMP THINKING TRICK)
if you want to skip the code block, you need "and" and make x False for the case you wanna skip and True for rest
if you want to run the code block, you need "or" and make x True for the case you wanna add and False for rest

SAFROT (SKIP AND FALSE RUN OR TRUE)

Now how to write this case   memory trick   
================================
### var1 T and var2 F  -- do something  
### var1 F and var2 T  -- do something  If opposites are happening do something 
### change it to be same and then use XOR equality 


if same is happening -- do something 
### reveal T and even T  -- do something  
### reveal F and even F  -- do something  

This is an XOR Gate (2 cases True out of 4)
just equate them cond1==cond2

================================

Matrix in python  ###INTERVIEW
==================================================
1. Creation :
arr1=[[0 for col in range(n)] for row in range(m)]   ###CORRECT
arr1=[[0]*n]*m #WRONG #DOESNT WORK PROPERLY BECAUSE LISTS ARE COPIES OF EACH OTHER
More grid logic

Here n is the number of cols, m is the number of rows
for i in range(m):
  for j in range(n):
# Note rows first and columns second, you iterate starting from the top row
# last element of grid is grid[-1][-1] or grid[m-1][n-1] 

Iteration : 
# In a symmetric matrix, you can traverse only till the diagonal by :
for i in range(m):
    for j in range(i,n,1):       #### This INCLUDES the diagonal numbers and upper diagonal

for i in range(m):
    for j in range(i):           ##### lower triangle ### excluding diagonal

### In a non symmetric matrix diagonal doesnt make much sense 

Swapping across diag1 i,j becomes j,i 
Swapping across diag2 i,j becomes n-1-i,n-1-j
--------------------------------------------------
accessing row and column in a matrix
rows
matrix[i] with give the ith row 
The limitation of a python 2d list that we cannot perform column-wise operations on a 2d list so we have a hack: REMEMBER
[row[i] for row in matrix] will give ith column 
=============================================
HEAP THEORY

As shown above, one approach is to make an additional empty list and push the elements into that list with heapq.heappush . The time complexity of this approach is O(NlogN) where N is the number of elements in the list.
A more efficient approach is to use heapq.heapify . Given a list, this function will swap its elements in place to make the list a min-heap. Moreover, heapq.heapify only takes O(N) time.

Creating max heap
----------------------------
numbers = [4,1,24,2,1]
# invert numbers so that the largest values are now the smallest
numbers = [-1 * n for n in numbers]
# turn numbers into min heap
heapq.heapify(numbers)

Remember!!! 
We want k minimum points. Which heap? max heap of size k (iterate through the whole thing)
We want k largest numbers. Which heap? min heap of size k 

How ? Like this
for key in dict1: # O(N)
    heappush(heap, (dict1[key], key)) # freq, item - O(log(k)) ##frequency goes first as that is the key on which heap is organised
	if len(heap)==k+1:  ## This is going to pop n-k times, whats left after this ? top K
	    heappop(heap)   ## popping removes minimum element out of k+1 because min heap by default

You want kth largest repeatedly? min heap of size k gives you this is O(1) time


==================================================
Different techniques we know and their tell-tale signs:
1. Binary search 
The ask is to find a O(logn) solution
2. Base explorer method
3. Bottom up dp with tabulation/Top down memoization
Brute force is exponential time, you are making "choices", feels like permutation combination,


5. Sliding window 
Usually we find an aggregate(min, max, count) over strings/arrays
Keywords: subarray,substring are used
6. Stacks 
7. DFS
8. BFS 
9. Recursion -- Trees,
10. Heap, Quickselect
Find kth maximum, kth min, top k etc

11. Union Find
a)detecting cycle in undirected graph
b)Making groups which have a common parent/ count no of unconnected graph components

12. Backtracking:
    Words used are subset, subsequence. Tree Structure is seen.



=============================================
Brute Force iteration on a list/string
=============================================

for i in range(len(nums)):
  for j in range(i+1,len(nums),1):  ### THIS WILL NOT INCLUDE THE ELEMENT ITSELF 

  for i in range(len(nums)):
  for j in range(i,len(nums),1):  ### THIS WILL INCLUDE THE ELEMENT ITSELF 

for i in range(len(nums)):
  for j in range(i):  ### THIS WILL NOT INCLUDE THE ELEMENT ITSELF

 for i in range(len(s)):
            for j in range(i+1):   ### THis will include the element 
==================================================
Swapping from the middle 
for k in range((j-i+1)/2):
    s[i+k],s[j-k]=s[j-k],s[i+k]  ## simple swap 
==================================================

#### STRINGS AND LISTS DO NOT GIVE INDEX ERROR(EMPTY) WHEN RANGE IS THERE OTHER THEY WILL GIVE ERROR
a=list("hello")
a[100:] #[] empty list and empty string 

List comprehension
[f(x) for x in sequence if condition]               ##filtering
[f(x) if condition else g(x) for x in sequence]
=============================================






==========================================================================================
657. Robot Return to Origin
There is a robot starting at position (0, 0), the origin, on a 2D plane. Given a sequence of its moves, judge if this robot ends up at (0, 0) after it completes its moves.
The move sequence is represented by a string, and the character moves[i] represents its ith move. Valid moves are R (right), L (left), U (up), and D (down). If the robot returns to the origin after it finishes all of its moves, return true.Otherwise, return false.
------------------------------------------------------------------------------------------
Logic: Just track the cordinates and check at the end if cordinates are zero or not.
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
Code1
 dict1={'U':[0,1],'D':[0,-1],'R':[1,0],'L':[-1,0]}
        
        x,y=0,0
        for i in range(len(moves)):
            x+=dict1[moves[i]][0]
            y+=dict1[moves[i]][1]
        
        if x==0 and y==0:
            return True 
        return False
==========================================================================================
Defanging IP adress 
Tags: String 
1. Strings in python are immutable. DONT TRY TO DO THIS IN PLACE.
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
Method1: Using itertion 
 res=''
        for i in range(len(address)):
            if address[i]=='.':
                res+='['
                res+=address[i]
                res+=']'
            else:
                res+=address[i]
        return res 

Method2: Using split and join 
list1=address.split('.')
return '[.]'.join(list1)
==========================================================================================
1047. Remove All Adjacent Duplicates In String
I simply convert string to list and check next with current if duplicated, i remove both and move pointer back one step
class Solution(object):
    def removeDuplicates(self, s):
        list1=list(s)
        
        i=0
        while i<=len(list1)-2:
            if list1[i]==list1[i+1]:
                list1.pop(i)
                list1.pop(i) ### DONT DO i+1 becuase after pop i+1 element automatically comes to I 
                if i-1<0:
                    i=-1
                else:
                    i=i-2
            i+=1
        

        return ''.join(list1)
        
Stack solution is worth looking into just see the infographic in solution. Why stack because next elements keep collapsing previous
------------------------------------------------------------------------------------------
class Solution(object):
    def removeDuplicates(self, s):
        stack=[]
        for i in range(len(s)):
            if len(stack)==0 or s[i]!=stack[-1]:
                stack.append(s[i])
            else:
                stack.pop()
        return ''.join(stack)
------------------------------------------------------------------------------------------

==========================================================================================
151. Reverse Words in a String
Tags: String 

list1=s.split(). ###cant use .split(' ') ##will not ignore whitespace
list1=list1[::-1]
return ' '.join(list1)
#O(N) -- both time and space 

==========================================================================================
186. Reverse Words in a String II
1. Tricky question without extra space 
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
## using extra space 

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        list1=''.join(s)
        
        list1=list1.split()
        list1=list1[::-1]
        list1=list(' '.join(list1))
        
        for i in range(len(s)):
            s[i]=list1[i]

 ## No extra space 
1. Rotate the whole list first 2. Then rotate each word 

class Solution(object):
    def reverse_string(self,i,j,s):  ### remember this fucntion 
        for k in range((j-i+1)/2):
            s[i+k],s[j-k]=s[j-k],s[i+k]
    
    def reverseWords(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        
        if len(s)==0:
            return 
        self.reverse_string(0,len(s)-1,s)
        
        i=0                           ### like base and explorer method 
        for j in range(len(s)):
            if s[j]==" ":
                self.reverse_string(i,j-1,s)
                i=j+1
        self.reverse_string(i,j,s)   ### the final word needs to be reverted
==========================================================================================
557. Reverse Words in a String III
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
class Solution(object):
    def reverseWords(self, s):
        list1=s.split()
        
        return ' '.join(x[::-1] for x in list1)
==========================================================================================
Encode and Decode TinyURL
I dont think this question wll be asked
string to integer
Bad question dont want to do it 
==========================================================================================
6. ZigZag Conversion
Tricky question 
------------------------------------------------------------------------------------------
Difficult question do again later,
------------------------------------------------------------------------------------------
1. Realize distance between peaks is constant 
2. Intermediate row values can be calculated from peak distances by adding and subtracting 
3. Answer is generated row by row and for each row you iterate through all top peaks 

class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if len(s)==0 or numRows==1:
            return s
        
        m=len(s)
        ans=''
        interval=2*numRows-2
        
        
        for i in range(numRows):
            j=0                  ### For each row I generate the peaks 
            while j-i<=m-1:      ### You cant write j<=m-1 because even if peak doesnt exist we still use that peak!!!
                p1=j-i           ### I calculate two points based on the peak 
                p2=j+i
                if i==0 or i==numRows-1:
                    if p1>=0:
                        ans+=s[p1]
                else:
                    if p1>=0 and p1<=m-1:   ### add both if they are within ranges 
                        ans+=s[p1]
                    if p2>=0 and p2<=m-1:
                        ans+=s[p2]
                j+=interval      ### For generating peak i increase the interval 
        return ans 
------------------------------------------------------------------------
def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if len(s)==0 or numRows==1:
            return s
        
        m=len(s)
        ans=''
        
        for i in range(numRows):
            peak=0
            
            while peak-i<=m-1:
                beforePeak=peak-i
                afterPeak=peak+i
                
                if i==0:
                    ans+=s[peak]
                elif i==numRows-1:
                    if afterPeak>=0 and afterPeak<=m-1:
                        ans+=s[afterPeak]
                        
                else:
                    if beforePeak>=0 and beforePeak<=m-1:    
                        ans+=s[beforePeak]
                    if afterPeak>=0 and afterPeak<=m-1:
                        ans+=s[afterPeak]
        
                peak+=2*numRows-2
        
        return ans 



==========================================================================================
Unique email address
Tags: String 
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
class Solution(object):
    def numUniqueEmails(self, emails):
        """
        :type emails: List[str]
        :rtype: int
        """
        dict1={}
        for i in range(len(emails)):
            res=''
            email=emails[i].split('@')[0]
            domain=emails[i].split('@')[1]
            for j in range(len(email)):
                if email[j]=='+':
                    break
                elif email[j]=='.':
                    pass
                else:
                    res+=email[j]
            if res+'@'+domain not in dict1:
                dict1[res+'@'+domain]=1
        return len(dict1.keys()) 
                    
==========================================================================================
804. Unique Morse Code Words
Tags: String 
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------

morse=[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        letters='abcdefghijklmnopqrstuvwxyz'
        morseDict=dict(zip(letters,morse))
        
        dict1={}
        for i in range(len(words)):
            word=words[i]
            res=''
            for j in range(len(word)):
                res+=morseDict[word[j]]
            if res not in dict1:
                dict1[res]=1
        return len(dict1)    
==========================================================================================
93. Restore IP Addresses
Given a string containing only digits, restore it by returning all possible valid IP address combinations.
Tags:
-------------------------------------------------------------------------------------------------------------------------
Logic: This sounds like a backtracking problem but can also be solved using simple iteration and brute force below
-------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
Done, Tricky do again
------------------------------------------------------------------------------------------

class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res=[]
        for i in range(1,4,1):   #### We only need to iterate till 4 places ## rest is useless 
            first=s[:i]
            if self.validIpChecker(first):
                for j in range(1,4,1):
                    second=s[i:i+j]  ### Getting the indexing right is critical
                    if self.validIpChecker(second):
                        for k in range(1,4,1):
                            third=s[i+j:i+j+k]
                            fourth=s[i+j+k:]
                            if self.validIpChecker(third) and self.validIpChecker(fourth):
                                ans='.'.join([first,second,third,fourth])
                                res.append(ans)
        return res
    def validIpChecker(self,str1):
        if len(str1)==0:
            return False
        if str1[0]=='0' and len(str1)>=2:
            return False
        if int(str1)<0 or int(str1)>=256:
            return False
        return True
==========================================================================================

728. Self Dividing Numbers
A self-dividing number is a number that is divisible by every digit it contains.
For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.
Also, a self-dividing number is not allowed to contain the digit zero.
-------------------------------------------------------------------------------------------------------------------------
Logic:
1.Pretty simple. Check for each number. Notice the use of else clause with the for loop.
2. Also notice we have to check for zero in the OR clause. 
Whenevr you see the OR clause think about who should be first 
-------------------------------------------------------------------------------------------------------------------------



class Solution(object):
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        ## logic
        ##
        ##
        res=[]
        for i in range(left,right+1,1):
            str1=str(i)
            for j in str1:
                if j=='0' or i%int(j)!=0:
                    break 
            else:
                res.append(i)
        
        return res   
==========================================================================================
+++++++++++++++++++++++++++++++++++
SLIDING WINDOW
+++++++++++++++++++++++++++++++++++

5. Longest Palindromic Substring

1.Brute Force- N2 to calculate  all the substrings and then N time to check each string if its a palindrome
O(N^3) 
2.If you expand from the center at each index, it only takes N time at each index. So overall O(N2) time
First deal with ODD PALINDROMES
a) iterate over each index, initialize two pointers j&k and start expanding them out until the condition is valid.
This will take care of ODD. For even, answer is simple do the same thing just j=i-1!!
Figured it out myself!!!


------------------------------------------------------------------------------------------
Done, Time limit exceeds with brute force 
------------------------------------------------------------------------------------------
# BRUTE FORCE
class Solution(object):
    def longestPalindrome(self, s):
        maxLen=1
        ans=s[0]
        for i in range(len(s)):
            for j in range(i):
                if self.isPalindrome(s[j:i+1]):
                    if len(s[j:i+1])>maxLen:
                        ans=s[j:i+1]
                        maxLen=max(maxLen,len(s[j:i+1]))
        return ans
                    
                            
    
    def isPalindrome(self,s):
        return s==s[::-1]
------------------------------------------------------------------------------------------
### EXPAND AROUND THE CENTRE ## TWICE LOOP FOR ODD AND EVEN

class Solution(object):
    def longestPalindrome(self, s):
        ans=""
        for i in range(len(s)):
            j=i
            k=i
            while j>=0 and k<=len(s)-1:
                if s[j]==s[k]:
                    j-=1
                    k+=1
                else:
                    break
            if (k-1)-(j+1)+1>len(ans):
                ans=s[j+1:k]
            ## same thing for even palindrome
            j=i-1
            k=i
            while j>=0 and k<=len(s)-1:
                if s[j]==s[k]:
                    j-=1
                    k+=1
                else:
                    break
            if (k-1)-(j+1)+1>len(ans):
                ans=s[j+1:k]
        return ans
==========================================================================================   
647. Palindromic Substrings
Same as the question before but we just keep a count, we ALLOW DUPLICATES!
class Solution(object):
    def countSubstrings(self, s):
        count=0
        for i in range(len(s)):
            j=i
            k=i
            while j>=0 and k<=len(s)-1:
                if s[j]==s[k]:
                    count+=1
                    j-=1
                    k+=1
                else:
                    break 
            j=i-1
            k=i
            while j>=0 and k<=len(s)-1:
                if s[j]==s[k]:
                    count+=1
                    j-=1
                    k+=1
                else:
                    break 
        return count


==========================================================================================   
3. Longest Substring Without Repeating Characters
Tag: Sliding Window

1. Brute force is O(n3) because n2 substrings and N to check each
------------------------------------------------------------------------------------------
2. Sliding window
What is the logic?
Is this like base explorer ? In base explorer, base sits while explorer explores. once explorer finds something, swapping happens.
Now base usually moves just one and explorer again moves.explorer ALWAYS moves.
This question is NOT BASE EXPLORER. why ? 
Here the front pointer stops and waits for the back pointer to explore and moves ahead only when things are valid again.
Inside the window things are valid always.
Common mistake is to snap i to j. DONT DO THIS MISTAKE.
Once you realize its sliding window its easy.

Time complexity: O(N) -->O(2n)=O(n). In the worst case each character will be visited twice by i and j

Space complexity : O(min(n, m)). We are using dictionary to save chars 
The size of the Set is upper bounded by the size of the string n and the size of the charset/alphabet m
------------------------------------------------------------------------------------------
I have provided many solutions, I try to remember standard pattern only.
------------------------------------------------------------------------------------------
ALGO: SLIDING_WINDOW_VIOLATION_CORRECTION 
a) We let the violation happen and then correct it

class Solution:
    def lengthOfLongestSubstring(self, s):
        dict1 = {}

        i = j = 0

        maxCount = 0
        while j <= len(s)-1:
            if s[j] not in dict1:
                dict1[s[j]]=1
            else:
                dict1[s[j]]+=1
            
            while dict1[s[j]] > 1:  ##condn for violation 
                dict1[s[i]] -= 1    ## correct the violation
                i += 1
            print(s[i:j+1])
            maxCount = max(maxCount, j - i + 1)

            j += 1
        return maxCount
------------------------------------------------------------------------------------------        
3. Sliding window optimized
the basic idea is, keep a hashmap which stores the characters in string as keys and their positions as values, and keep two pointers which define the max substring. move the right pointer to scan through the string , and meanwhile update the hashmap. If the character is already in the hashmap, then move the left pointer to the right of the same character last found. Note that the two pointers can only move forward.
We can further optimize this, by moving i to the last known index of the violation. Smart!!

Only two cases either there is a violation or isnt
Why are these cases so painful?

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        dict1={}
        countMax=0
        i=0
        j=0
        while j<=len(s)-1:  
            if s[j] not in dict1 or dict1[s[j]]<i or dict1[s[j]]>j: ## we increase j if there is no violation 
                                                                    ## if last seen is outside of [i,j] window its ok
                dict1[s[j]]=j
                countMax=max(countMax,j-i+1)    ## dont reverse the order of checking this and incrementing j
                j+=1
            else:
                i=dict1[s[j]]+1                 ## moving i to the next element of last seen
                                                ## we did additional processing here because the strict wondow condition wont work 
                dict1[s[j]]=j
                countMax=max(countMax,j-i+1)    ## dont reverse the order of checking this and incrementing j
                j+=1
                

                       
        return countMax 
==========================================================================================   
340. Longest Substring with At Most K Distinct Characters
159. Longest Substring with At Most Two Distinct Characters
This is similar to the previous question 
My loop structure was bad so we will use this loop structure where addition on j happens first and then we check validity of window.

ALGO: SLIDING_WINDOW_VIOLATION_CORRECTION

class Solution:
    def lengthOfLongestSubstringKDistinct(self, s, k):
        if k==0:
            return 0
        dict1 = {}
        i = j = 0
        maxCount = 0
        while j <= len(s)-1:
            if s[j] not in dict1:
                dict1[s[j]]=1
            else:
                dict1[s[j]]+=1
            
            while len(dict1.keys())>k:
                if dict1[s[i]]==1:
                    del dict1[s[i]]
                else:
                    dict1[s[i]]-=1
                i += 1
            #print(s[i:j+1])
            maxCount = max(maxCount, j - i + 1)

            j += 1
        return maxCount
------------------------------------------------------------------------------------------
Further optimization:We can increase i 1 by 1 or be smarter about it again. For example: LOVELEETCODE & k=4
When you reach T, O can be directly removed because thats the lowest in the dictionary values.
------------------------------------------------------------------------------------------
class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        if k==0:
            return 0
        dict1={}
        maxCount=0
        i=0
        j=0
        while j<=len(s)-1:
            dict1[s[j]]=j                       ### we keep updating with the rightmost index, we only care about this
            if len(dict1.keys())<=k :           ##non violation condn 
                maxCount=max(maxCount,j-i+1)
            else:
                minm=min(dict1.values())
                i=minm+1
                del dict1[s[minm]]
            j+=1
        
        return maxCount
========================================================================================== 
Longest Repeating Character Replacement
Same as before,let violation of dictionary happen but we dont record.

1. I used the window acceptance criteria as this: sum(dict1.values())-max(dict1.values())<=k
Why this? I want to remove the most frequent charcter and replace the rest. Is their sum less than k? If yes we can go on extending our window.
Now, instead of sum(dict1.values) you can also do j-i+1. Both are the same thing

ALGO: SLIDING_WINDOW_VIOLATION_CORRECTION 
class Solution:
    def lengthOfLongestSubstring(self, s):
        dict1 = {}

        i = j = 0

        maxCount = 0
        while j <= len(s)-1:
            if s[j] not in dict1:
                dict1[s[j]]=1
            else:
                dict1[s[j]]+=1
            
            while dict1[s[j]] > 1:
                dict1[s[i]] -= 1
                i += 1
            print(s[i:j+1])
            maxCount = max(maxCount, j - i + 1)

            j += 1
        return maxCount
========================================================================================== 
438. Find All Anagrams in a String
ALGO: SLIDING_WINDOW_VIOLATION_CORRECTION

class Solution:
    def findAnagrams(self, s, p):
        dict1=collections.Counter(p)
        dict2 = {}
        i = j = 0
        ans=[]
        while j <= len(s)-1:
            if s[j] not in dict2:
                dict2[s[j]]=1
            else:
                dict2[s[j]]+=1
            
            while j-i+1>len(p):  ##violation condition 
                if dict2[s[i]]==1:
                    del dict2[s[i]]
                else:
                    dict2[s[i]]-=1
                i += 1
            #print(s[i:j+1])
            if dict2==dict1:
                ans.append(i)

            j += 1
        return ans



==========================================================================================   
6.Minimum Window Substring
What is supposed to happen?

Double while loop pattern is the key. Almost killed myself coding in other patterns
ALGO: SLIDING_WINDOW_VIOLATION_CORRECTION
class Solution(object):
    def minWindow(self, s, t):
        dict1=collections.Counter(t)
        dict2 = {}
        i = j = 0
        minM=float("inf")
        ans=''      ### for edge cases
        while j <= len(s)-1:
            if s[j] not in dict2:
                dict2[s[j]]=1
            else:
                dict2[s[j]]+=1
            
            while all([dict2[key]>=dict1[key] if key in dict2 else False for key in dict1.keys()]):  ##violation condition
                if j-i+1<minM:
                    ans=s[i:j+1]
                    minM=j-i+1
                if dict2[s[i]]==1:
                    del dict2[s[i]]
                else:
                    dict2[s[i]]-=1
                i += 1
            #print(s[i:j+1])
            
            j += 1
        return ans
==========================================================================================   
992. Subarrays with K Different Integers

This problem will be a very typical sliding window,
if it asks the number of subarrays with at most K distinct elements.
Just need one more step to reach the folloing equation:
exactly(K) = atMost(K) - atMost(K-1)

class Solution(object):
    def subarraysWithKDistinct(self, nums, k):
        return self.subarraysWithAtMostKDistinct(nums,k)-self.subarraysWithAtMostKDistinct(nums,k-1)
    
    def subarraysWithAtMostKDistinct(self, nums, k):
        dict1={}
        i = j = 0
        count=0
        while j <= len(nums)-1:
            if nums[j] not in dict1:
                dict1[nums[j]]=1
            else:
                dict1[nums[j]]+=1
            
            while len(dict1)>k:    ##violation condition 
                #print("contracting",nums[i:j+1])
                if dict1[nums[i]]==1:
                    del dict1[nums[i]]
                else:
                    dict1[nums[i]]-=1
                i += 1
            
            #print(nums[i:j+1])
            count+=j+1-i            ### HUGE EXPLANATION FOR THIS!!!
            #print(count)
            j += 1
        return count

https://leetcode.com/problems/subarrays-with-k-different-integers/discuss/523136/JavaC%2B%2BPython-Sliding-Window

ret = total number of contiguous subarrays with at most K different integers.
j - i + 1 = length of each valid contiguous subarray with at most K different integers.

Recall that to get the the total number of combinations of contiguous subarrays of an array of length n, we can do the summation of 1 + 2 + ... + (n - 1) + n.

In the context of this problem, the code above will produce valid (sliding) windows.
For example A = [1,2,1,2], K = 2 then windows are [1], [1,2], [1,2,1], [1,2,1,2].

As we expand the length of that window, we can sum the length of those windows to get a count of our different combinations, because if our "complete" window was [1,2,1,2],
we could do 1 + 2 + 3 + 4 (or length of [1] + length of [1,2] + length of [1,2,1] + length of [1,2,1,2]).

We also noticed that if instead A = [1,2,1,2,3], K = 2 (added an additional 3 at the end of the array).

The sliding window did not return [2] (second 2 in the array) because the window expanded to [1,2,1,2,3] -- invalidating the window and then compressed the window to [2,3] by moving i forward. This allowed us to skip those duplicate subarrays. You can expand this to other examples including where K is larger. The fact that our sliding window compresses by moving forward i will allow the lower summations to be ignored (i.e. our duplicate subarrays) that would be captured by summing the windows before it.


==========================================================================================   
239. Sliding Window Maximum
Time complexity: O(N*k) - TLE
1. Simply check max at each window
Our famous double while loop again

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        #dict2={}
        i=0
        j=0
        ans=[]
        while j<=len(nums)-1:
            while j-i+1>k:                         ## violation condition
                i+=1
            if j-i+1==k:
                ans.append(max(nums[i:j+1]))
            
            j+=1

        return ans
        
2. Simply check max at each window, but using max heap
Time complexity: O(N*log(k))


class Solution(object):
    def maxSlidingWindow(self, nums, k):
        #dict2={}
        i=0
        j=0
        ans=[]
        heap=[]
        while j<=len(nums)-1:
            heapq.heappush(heap,(-1*nums[j],j))       ## we create a max heap of k elements
            ans.append(heap[0][0]*-1)                 ## we get max without pop
            
            while j-i+1>k:                      ## violation condition ## heap has k+1 elements
                heapq.heappush(heap, (-1*nums[i+k-1], i+k-1))
                i+=1
            if j-i+1==k:
                ans.append(heapq.heappop(heap)) ## we remove the highest element)
            
            j+=1

        return ans

==========================================================================================   
567. Permutation in String
Two type window moving which is based on violation condition

1. Violated immediately after length is more
class Solution(object):
    def checkInclusion(self, s1, s2):
        dict1=collections.Counter(s1)
        dict2={}
        i=0
        j=0
        while j<=len(s2)-1:
            if s2[j] in dict1:
                if s2[j] not in dict2:                    ## add to dict1
                    dict2[s2[j]]=1
                else:
                    dict2[s2[j]]+=1
            
            while j-i+1>len(s1):
                if s2[i] in dict2:                        ## violation condition
                    if dict2[s2[i]]==1:                   ## correction of violation
                        del dict2[s2[i]]
                    else:
                        dict2[s2[i]]-=1
                i+=1
            if dict1==dict2:                    ## what is a better way to do this comparision?
                return True
            j+=1

        return False
2. Violated when we find abc and then contract. All violated strings are valid strings, we just need to see if its length is equal to len s2
class Solution(object):
    def checkInclusion(self, s1, s2):
        dict1=collections.Counter(s1)
        dict2={}
        i=0
        j=0
        while j<=len(s2)-1:
            if s2[j] in dict1:
                if s2[j] not in dict2:                    ## add to dict1
                    dict2[s2[j]]=1
                else:
                    dict2[s2[j]]+=1
            
            while all([dict2[key]>=dict1[key] if key in dict2 else False for key in dict1.keys()]):
                if j-i+1==len(s1):
                    return True
                if s2[i] in dict2:                        ## violation condition
                    if dict2[s2[i]]==1:                   ## correction of violation
                        del dict2[s2[i]]
                    else:
                        dict2[s2[i]]-=1
                i+=1
            j+=1

        return False
3. More optimal way to compare dictionaries? Should be useful across all these problems
==========================================================================================   

400. Nth Digit
==========================================================================================   
## brute forcing by simply adding up len strings to count, breaking and then iterating chars on the word
class Solution(object):
    def findNthDigit(self, n):
        i=0
        count1=0
        while True:
            i+=1
            count2=count1+len(str(i))
            if count2<n:
                count1=count2
            else:
                break
        return str(i)[n-1-count1]
------------------------------------------------------------------------------------------
1-------9 9*1 = 9 digits
10-----99 90 *2 = 180 digits
100---999 900 * 3 = 2700 digits
Now, for example gave N = 1000, then 1000-9-180 = 811, it means the 811th digit local in [100, 999], and we know each number like 100 has three digit, so 811 / 3 = 270,
Then, we know the 270th number in [100, 999], is 270th + 100 (start from 100) = 370.
370 still has three digit, which one is the answer? 3, 7, 0
811 % 3 = 1, so the first one is the answer, so return 3.

class Solution(object):
    def findNthDigit(self, n):
        digit=1
        base=9
        while n>digit*base:
            n-=digit*base
            digit+=1
            base=base*10
        number_where_ans_will_be_found=(10**(digit-1))-1+int(math.ceil(float(n)/digit))
        ### why this because fractions need to go be rounded up and not down, this will avoid edge cases
        ### how do we round fractions up instead of down int(math.ceil(float(num)/denominator))
        ### REMEMBER
        index_where_ans_will_be_found=(n)%digit-1       ## WHY THE -1 , because of 0 indexing
        return str(number_where_ans_will_be_found)[index_where_ans_will_be_found]




==========================================================================================   
172. Factorial Trailing Zeroes
Tags: Math
------------------------------------------------------------------------------------------
Done, Tricky have to Remember
------------------------------------------------------------------------------------------

Logic:
Explanation:
The ZERO comes from 10.
The 10 comes from 2 x 5
And we need to account for all the products of 5 and 2. likes 4ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â5 = 20 ...
So, if we take all the numbers with 5 as a factor, we'll have way more than enough even numbers to pair with them to get factors of 10

Another way of thinking about this is just divide the number by 5 and then keep on dividing the quotient too and add all the quotients for your ANSWER !!!!!!
25/5=5    5/5=1    5+1=6 Ans
100/5=20 20/5=4  20+4 Ans 
125/5=25 25/5=5 5/5=1 25+5+1 Ans 
Basically how this works is till 25, 5 multiples of 5 exist + 1 more because 25 has extra 
-------------------------------------------------------------------------------------------------------------------------
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        sum1=0
        while n>0:
            sum1+=n/5
            n=n/5
        return sum1

==========================================================================================  
319. Bulb Switcher
Tags: Math
------------------------------------------------------------------------------------------
Done, Tricky have to Remember
------------------------------------------------------------------------------------------
Logic:
Initial state of bulbs is off.
Bulb at position 6 will be toggled 4 times. 1,2,3,6
Bulb at position 7 will be toggled 2 times. 1,7
Bulb at position 4 will be toggled 3 times only!!!   1,2,2,4 ---> 1,2,4 
k is all factors (including 1) for a given bulb at location i  
if k is even - retrn to initial(off) , odd bulb on.
If you take any number other than perfect squares, number of factors is odd! why ? because factor involves 
two numbers both smaller than the number. 

So only in the case of perfect squares, bulb will be moved from initial state(off) to ON
How many perfect sqaures exist smaller than N?? 
simple floor of square root of N because all smaller squares are within this N 
------------------------------------------------------------------------------------------
return int(n**0.5)
==========================================================================================
11. Container With Most Water
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
------------------------------------------------------------------------------------------
Done, Tricky have to Remember
------------------------------------------------------------------------------------------
*** Maximum area will not be always between first maximum and second maximum numbers. Because, it is possible that there are two lower numbers having more distance between them so these numbers will give height area.
Use two pointers. (left, right)
Keep track of maxArea so far. 
Try to maximize the area. So, always move pointer which has smaller pole. Because bottleneck is the smaller one!!! (in hopes of finding a longer pole)
It is NOT possible to get a bigger area by moving the bigger length? Why ? Because even if you find a 
bigger pole (than our current). min height is still same but length decreased so less area.
 If you move the smaller one it is possible that you find a length that can compensate for the 
 decreased distance. So we always move the minimum !!!
What to do when both are equal ??? move anyone. Because we are sure that a different  answer is 
only possible if both new lengths are bigger than the equal ones. 
So if either pointer reaches the big ones in the middle it will be just stuck there 
until the other one doesnÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢t reach its correct position. 
Is it possible to have a bigger area later which we canÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢t get using this greedy approach ?
no took examples and checked. 
------------------------------------------------------------------------------------------

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        ##logic 
        #[1,8,6,2,5,4,8,3,7]
        
        
        i=0
        j=len(height)-1
        
        maxArea=0
        while i<j:
            maxArea=max(maxArea,(j-i)*min(height[i],height[j]))
            if height[i]<height[j]:
                i+=1
            else:
                j-=1
        return maxArea
==========================================================================================
Jewels and Stones
Tags: Dict
------------------------------------------------------------------------------------------
Done
------------------------------------------------------------------------------------------
You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, so "a" is considered a different type of stone from "A".

Example 1:
Input: J = "aA", S = "aAAbbbb"
Output: 3
------------------------------------------------------------------------------------------
class Solution(object):
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        dict1={}
        
        for i in range(len(J)):
            if J[i] not in dict1:
                dict1[J[i]]=1
        
        count=0
        for j in range(len(S)):
            if S[j] in dict1:
                count+=1
        return count


==========================================================================================
1007. Minimum Domino Rotations For Equal Row
Tags: 
------------------------------------------------------------------------------------------
Done, slightly tricky
------------------------------------------------------------------------------------------
Logic: 
There are 4 possible cases:
make values in A equal to A[0]
make values in B equal to A[0]
make values in A equal to B[0]
make values in B equal to B[0]
For each case we count rotations and we get the min rotations among them.
NO NEED TO ACTUALLY SWAP!!!!!

 ------------------------------------------------------------------------------------------
class Solution(object):
    def minDominoRotations(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        min4= min(self.swaptoTarget(A[0],A,B),
                  self.swaptoTarget(A[0],B,A),
                  self.swaptoTarget(B[0],A,B),
                  self.swaptoTarget(B[0],B,A))
        
        if min4==float('inf'):
            return -1 
        else:
            return min4 
    def swaptoTarget(self,target,row1,row2):
        rotations=0
        for i in range(len(row1)):
            if row1[i]!=target:
                if row2[i]==target:
                    rotations+=1
                else:
                    rotations=float('inf')
        return rotations

==========================================================================================
609. Find Duplicate File in System
Given a list of directory info including directory path, and all the files with contents in this directory, you need to find out all the groups of duplicate files in the file system in terms of their paths.

A group of duplicate files consists of at least two files that have exactly the same content.
Tags: String
------------------------------------------------------------------------------------------
Done
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
Logic: 
mostly string splitting logic and then using a dict to store content so that we know when it repeats 
 ------------------------------------------------------------------------------------------
class Solution(object):
    def findDuplicate(self, paths):
        """
        :type paths: List[str]
        :rtype: List[List[str]]
        """
        dict1={}
        for i in range(len(paths)):
            path=paths[i].split()[0]
            filelist=paths[i].split()[1:]
            for j in range(len(filelist)):
                file,content=filelist[j].split('(')
                if content not in dict1:
                    dict1[content]=[path+'/'+file]
                else:
                    dict1[content].append(path+'/'+file)
        
        res=[]
        for k in list(dict1.keys()):
            if len(dict1[k])>1:
                res.append(dict1[k])
        return res 
==========================================================================================            


++++++++++++++++++++++++++++++++
+++++++Group : STACK+++
++++++++++++++++++++++++++++++++

TWO THINGS 
1. DOUBLE WHILE VIOLATION CORRECTION 
2. IT IS POSSIBLE TO WRITE THIS THE BELOW IF ELSE OTHER WAY AROUND!
   BUT I DONT! WHY I GIVE CONDITIONS TO THE PART WHERE WE POP 
   AND I CHECK STACK EMPTY CONDITION HERE! THIS WILL MAKE LIFE EASIER ALWAYS
   stack and stack[-1]==something


20. Valid Parentheses
Done 
######## whenever you find a left bracket append to stack. if you find a right bracket search for corresponding 
##left bracket in the stack and if found pop . if not found is not valid. 
## SOURCE OF ERROR IS NOT CHECKING FOR LIST EMPTY CONDITION 

-------------------------------------------------------------------------------------------
class Solution(object):
    def isValid(self, s):
        dict1= {")":"(","]":"[","}":"{"}
        
        i=0
        stack = []
        while i <= len(s)-1:
            if s[i] not in dict1:
                stack.append(s[i])
            else:
                ### IT IS POSSIBLE TO WRITE THIS THE BELOW IF ELSE OTHER WAY AROUND!
                ### BUT I DONT! WHY I GIVE CONDITIONS TO THE PART WHERE WE POP 
                ### AND I CHECK STACK EMPTY CONDITION HERE! THIS WILL MAKE LIFE EASIER ALWAYS
                
                if stack and stack[-1] == dict1[s[i]]:  ##SAFROT
                    stack.pop()
                else:
                    return False
            i += 1
            
        if not stack:
            return True
        else:
            return False
Note : This is a simple question and gave me more pain in Stack_violation_correction ALGO
==========================================================================================
71. Simplify Path
Simple question doesnt deserve medium 
class Solution(object):
    def simplifyPath(self, path):
        pathList=path.split("/") ### even double slashes are resolved using this we just have more empty strings
        stack=[]
        for i in range(len(pathList)):
            if pathList[i]=="..":
                if len(stack)!=0:       ### in stack you always need to check if stack is not empty before pop
                    stack.pop()
            elif pathList[i]=="." or pathList[i]=="":
                pass
            else:
                stack.append(pathList[i])
        
        return "/"+"/".join(stack) 
==========================================================================================
921. Minimum Add to Make Parentheses Valid
Here we need to remove(pop out) all the valid ones and then what is remaining has to be dealt with 
when we get ( we can only hope to find complement later so we always add it 
when we get ) 2 cases : either we pop ( it out. or add it to the queue. Here we already know that this cant be really dealt it but it has to be counted.

So the final answer has ) which cant be dealt with and ( which is waiting for partner. Hopeless and hopeful

Answer is length of the remaining queue because for each of them you have to add extra

class Solution(object):
    def minAddToMakeValid(self, s):
        stack =[]
        for i in range(len(s)):
            if s[i]=="(":
                stack.append(s[i])
            elif s[i]==")":
                if stack and stack[-1]=="(":
                    stack.pop()
                else:
                    stack.append(s[i]) ## we add hopeless to stack just to maintain count, we can do this otherwise too
        return len(stack)
=========================================================================================                      
1249. Minimum Remove to Make Valid Parentheses
The difference is here we need the string too. we keep adding hopefuls and removing hopeless.
In the second pass, we remove hopefuls which didnt find a partner, we start from the back to get these


class Solution(object):
    def minRemoveToMakeValid(self, s):
        list1=[]
        stack =[]
        for i in range(len(s)):
            if s[i]=="(":
                stack.append(s[i])
                list1+=s[i]
            elif s[i]==")":
                if stack and stack[-1]=="(":    ## this is where a hopeful pops out
                    stack.pop()
                    list1+=s[i]
                else:
                   pass      ## this is case where ) comes this is hopeless ## we dont add to stack or list because its hopeless
                
            else:
                list1+=s[i]
        
        m=len(stack)            ## search for hopefuls which didnt find their match 
        for i in range(len(list1)-1,-1,-1):
            if m==0:
                return ''.join(list1)
            elif list1[i]=="(":
                list1.pop(i)
                m-=1
        return ''
o(n) time and o(1) space
 ------------------------------------------------------------------------------------------
 Another way to do this is keeping track of indexes of hopefuls so that we dont have to search in the second pass. 
 The problem is we will need to add hopeless too to the list because index is count sensitive.
 Now the issue is to sepearate hopeless ) with matched ) so we change hopeless ) to . 
 
 class Solution(object):
    def minRemoveToMakeValid(self, s):
        list1=[]
        stack =[]
        for i in range(len(s)):
            if s[i]=="(":
                stack.append(i)
                list1+=s[i]
            elif s[i]==")":
                if stack:
                    stack.pop()
                    list1+=s[i]
                else:
                    list1+="."
                    
            else:
                list1+=s[i]
                    
                
            
        
        for i in range(len(stack)-1,-1,-1):  ## removing the hopefuls which didnt get lovers
            list1.pop(stack[i])
        return ''.join([x for x in list1 if x!="."]) ### removing the hopeless which we marked using .
==========================================================================================

==========================================================================================
1111. Maximum Nesting Depth of Two Valid Parentheses Strings
The language of the question is unnecessarily complex to understand, what you need to do is simple.
I need to break down the the given Valid parenthesis into two valid parenthesis in a way i make them as less nested as possible. So I just do this in a greedy way. 
I can either get a ( or ). I maintain a count for ( for both stacks. I will always assign new ( to the lesser count stack as I dont want nesting and I will assign new ) to greater count stack as I want to finish off exising (.
SEEKING IMMEDIATE REWARD AND NOT CARING IF THIS IS THE OPTIMAL.THATS WHY GREEDY. THATS IT!!

Honestly a decision tree can be formed because there are CHOICES. but greedy gives the answer based on intuition.

class Solution(object):
    def maxDepthAfterSplit(self, seq):
        """
        :type seq: str
        :rtype: List[int]
        """
        stackA=[]
        countA=0
        stackB=[]
        countB=0
        ans=[None for i in range(len(seq))]
        for i in range(len(seq)):
            if seq[i]=="(":
                if countA<=countB:          ### give it to the lesser count 
                    stackA.append(seq[i])
                    countA+=1
                    ans[i]=0
                else:
                    stackB.append(seq[i])
                    countB+=1
                    ans[i]=1
            elif seq[i]==")":               ### give it to the greater count 
                if countA<=countB:
                    stackB.append(seq[i])
                    countB-=1
                    ans[i]=1
                else:
                    stackA.append(seq[i])
                    countA-=1
                    ans[i]=0
    
        return ans 
==========================================================================================
696. Count Binary Substrings

## stack solution TLE  -- O(N2)
For every index, there can only be one answer, so i check each index
i tried to use stack here, but simple stack isnt the answer

"001011" --> will give wrong answer according to my stack logic. why pop and pushes need to be together. do we really need stack


class Solution(object):
    def countBinarySubstrings(self, s):
        i=0
        count1=0
        while i<=len(s)-1:
            j=i
            stack=[]
            counter="push"
            while len(stack)==0 or j<=len(s)-1:
                
                if j!=i and len(stack)==0:
                    #print(s[i:j])
                    count1+=1
                    break
                elif len(stack)!=0 and ((s[j]=="0" and stack[-1]=="1") or (s[j]=="1" and stack[-1]=="0")):
                    stack.pop()
                    counter="pop"
                elif counter=="pop": ## i modify stack logic to break if consecutive pops dont happen
                    break
                else:
                    stack.append(s[j])
                j+=1
            i+=1
        return count1 
 ------------------------------------------------------------------------------------------
## The same logic can be achieved by using counters and count -- TLE again -- O(N2)
class Solution(object):
    def countBinarySubstrings(self, s):
        i=0
        count1=0
        while i<=len(s)-1:
            j=i
            start=s[j]
            count2=0
            counter=True
            while count2==0 or j<=len(s)-1:
                if j!=i and count2==0:
                    #print(s[i:j])
                    count1+=1
                    break
                elif counter==True and s[j]==start:
                    count2+=1
                elif counter==False and s[j]==start:
                    break
                else:
                    count2-=1
                    counter=False
                j+=1
            i+=1
        return count1    
 ------------------------------------------------------------------------------------------
Fucking stupid and erratic solution:
https://leetcode.com/problems/count-binary-substrings/discuss/108625/JavaC%2B%2BPython-Easy-and-Concise-with-Explanation

First, I count the number of 1 or 0 grouped consecutively.
For example "0110001111" will be [1, 2, 3, 4].

Second, for any possible substrings with 1 and 0 grouped consecutively, the number of valid substring will be the minimum number of 0 and 1.
For example "0001111", will be min(3, 4) = 3, ("01", "0011", "000111")

class Solution(object):
    def countBinarySubstrings(self, s):
        
        groups=[]
        count=1
        for i in range(1,len(s),1):
            if s[i]==s[i-1]:
                count+=1
            else:
                groups.append(count)
                count=1
        groups.append(count) ## end of array append
        
        sum1=0
        for i in range(1,len(groups),1):
            sum1+=min(groups[i],groups[i-1])
            
        return sum1
Group: Substrings
==========================================================================================
    

Asteroid Collisions
==========================================================================================
Tags: Stack, Monotonic
Done, did it on my own 

------------------------------------------------------------------------------------------
Logic : question very similar to "valid parenthesis" and more similar to daily temp. Why we compare each element to sometime on the top of the stack
1. See any element add it to the stack
2. 4 posibilties 
a) stack+ new+      add to stack 
b) stack+ new-      collide, stack+ bigger: move to next new, new- bigger (new pointer stays but element on stack removed), if equal stack last element destroyed and pointer moves ahead
c) stack- new+      add to stack 
d) stack- new-      add to stack 
e) Edge cases : stack empty -- add to stack 

collapse these cases into code 

Think till when we wanna do this, till our pointer reaches end of list 
------------------------------------------------------------------------------------------  
class Solution(object):
    def asteroidCollision(self, asteroids):
        stack1=[]
        i=0
        while i<=len(asteroids)-1:
            if len(stack1)==0 or asteroids[i]>0 or (asteroids[i]<0 and stack1[-1]<0):
                stack1.append(asteroids[i])
                i+=1
            elif asteroids[i]*-1>stack1[-1]:
                    stack1.pop()
            elif asteroids[i]*-1==stack1[-1]:
                    stack1.pop()
                    i+=1
            else:
                i+=1
        return stack1
------------------------------------------------------------------------------------------ 
SHITTY CODE !!! FINAL IS THE NEXT ONE
mine more intuituve sense and logical (MONOTONIC STACK STANDARD PATTERN)
notice 2 while loops in a stack
first while loop is for greater array and 2nd is for stack operation
HERE WE HAVE TO UNRAVEL THE BOTTOM ON VIOLATION, SO WE DONT LET VIOLATION HAPPEN


class Solution(object):
    def asteroidCollision(self, asteroids):
        stack=[asteroids[0]]
        m=len(asteroids)
        
        i=1
        while i<=m-1:
            current=asteroids[i]
            while stack and current<0 and stack[-1]>0:   ## collision only in one case ##easy to forget stack empty case
                popped=stack.pop()  ###postive number 
                if abs(popped)>abs(current):  ##negative number stopped
                    current=popped      ## we put back again 
                elif abs(popped)<abs(current): ##negative number keeps on popping
                    pass
                else:
                    current=None    ## we put nothing back
                    break
            if current:
                stack.append(current)
    
            i+=1
        return stack
------------------------------------------------------------------------------------------ 
MODEL OF PERFECTION !! THE STANDARD CODE WHERE WE LET VIOLATION HAPPEN AND THEN WE DEAL WITH IT!!
WE GOT OUR TEMPLATE FOR STACK !!!!!!
ALGO_NAME: Stack_violation_correction

class Solution(object):
    def asteroidCollision(self, asteroids):
        stack=[]
        m=len(asteroids)
        
        i=0
        while i<=m-1:
            stack.append(asteroids[i])
            #print(stack)
            while len(stack)>=2 and stack[-2]>0 and stack[-1]<0:    ### NOTICE NO i inside the while loop
                #print(stack)
                if abs(stack[-1])<abs(stack[-2]):  ##negative number stopped
                    stack.pop() ##
                elif abs(stack[-1])>abs(stack[-2]): ##negative number keeps on popping
                    stack.pop(len(stack)-2)         ## We popped the positive
                else:
                    stack.pop()         ##popped both 
                    stack.pop()         ### be careful with consecutive pops
    
            i+=1
        return stack
==========================================================================================


Like in the case of daily temp, a negative number pops until it finds a negative or gets destroyed
In daily temp , a larger number keeps popping smaller ones.

THEORY TIME
The pattern here is similar to SLW, 
while loop for outside , 
while violation condition, 
correction 
pretend as if no violation and add to stack
THEORY TIME
==========================================================================================
636. Exclusive Time of Functions
We need to maintain a stack ofcourse, and we also need to keep track of prev time using which we update how much has passed since last update.
Tricky part is when end happens, prevtime is extra by 1. because end is reported at 5th sec actually happens at end of 5th.
So prevTime and time update both increase by 1

class Solution(object):
    def exclusiveTime(self, n, logs):

        dict1={x:0 for x in range(n)}
        stack=[]
        for i in range(len(logs)):
            fn,state,time=logs[i].split(":")
            fn,time=int(fn),int(time)
            if state=="start":
                if stack:
                    dict1[stack[-1]]+=time-prevTime
                stack.append(fn)
                prevTime=int(time)
                
            else:
                dict1[stack.pop()]+=time-prevTime+1 ## end update has extra 1 here too 
                
                prevTime=time+1 ### end update has extra 1 
                
        
        return [dict1[key] for key in sorted(dict1.keys())]
==========================================================================================
150. Evaluate Reverse Polish Notation
Integer Division is Floor division in Python 2. Lol. I never knew this.
2/5 =0 but -2/5=-1 (took the floor each time)
Now in this question, we want Division between two integers to truncate towards zero
For this to happen, convert to float and back to int


ALGO_NAME: Stack_violation_correction

class Solution(object):
    def evalRPN(self, tokens):
        ops=["+","-","*","/"]
        i=0
        stack=[]
        while i<=len(tokens)-1:
            stack.append(tokens[i])
            #print(stack)
            while stack and stack[-1] in ops:                                    ##violation 
                operator=stack.pop()                                             ##correction
                num1=int(stack.pop())
                num2=int(stack.pop())       ### be careful with consecutive pops
                if operator=="+":
                    new=num2+num1
                elif operator=="-":
                    new=num2-num1
                elif operator=="*":
                    new=num2*num1
                elif operator=="/":
                    new=int(float(num2)/num1)
                stack.append(new)
                
            i+=1
        return stack[-1]

1.I used the double while, violation correction method which i got from SLW
2.I didnt necessarily have to check for empty stack here, because empty stack is impossible, but I did out of format conforming
==========================================================================================
Largest Rectangle in a Histogram , 
Tags: Stack 
------------------------------------------------------------------------------------------
HARD QUESTION
Brute Force1 : Simple iteration ON3
------------------------------------
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # if len(heights)==1:
        #     return heights[0]
        
        max1=0
        for i in range(len(heights)):
            for j in range(i,len(heights),1):
                min1=float("inf")
                for k in range(i,j+1,1):
                    min1=min(min1,heights[k])
                area= (j-i+1)*min1
                max1=max(area,max1)
        return max1

Brute Force2 : Simple iteration ON2
--------------------------------------
we can find the bar of minimum height for current pair by using the minimum height bar of the previous pair.
In mathematical terms, minheight=min(minheight, heights(j))

class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # if len(heights)==1:
        #     return heights[0]
        
        max1=0
        for i in range(len(heights)):
            min1=float("inf")           ################ We are initializing minimum for each i
            for j in range(i,len(heights),1):
                min1=min(min1,heights[j])   ###### Then updating the minimum as we iterate through j by using prev min 
                area= (j-i+1)*min1
                max1=max(area,max1)
        return max1


THONK OF THIS AS DAILY TEMP BUT DAILY TEMP HAS DECREASING STACK AND A LARGER ELEEMNT POPS ALL THE PREV 
HERE ITS INCREASING



Logic :
https://www.youtube.com/watch?v=VNbkzsnllsU&t=413s
Question is very difficult
Some points 
1. the stack is used to stored the index.
2. when we pop the stack, the heights[stack.pop()] is monotonically decreasing, and therefore we can treat the original index (i - 1) as one anchor point to calculate the width.
|
| |
| |  |  +++  new element (at i)

popped area ==  (stack[-1]-stack[-2]-1) * height[stack[-1]]  

3. After we finish the list the stack is monotonically increasing, so now 
area = (len(heights)-stack[-2]-1)* stack[-1]

### the rectangle starts from len height and base is at stack[-2] as the mid elements were greater than stack[-1]	 


    |
 |  |
||  |   
------------------------------------------------------------------------------------------
Code:

class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        
        stack1=[-1]
        max1=0
        i=0
        while i<len(heights):
            if len(stack1)==1:
                stack1.append(i)
                i+=1
            elif heights[i]<heights[stack1[-1]]:
                popped=stack1[-1]
                area=(i-1-stack1[-2])*heights[popped]   ########## THIS EQUATION IS DIFFICULT TO COME UP WITH    
                max1=max(max1,area)
                stack1.pop()
            else:
                stack1.append(i)
                i+=1
        
        while stack1[-1] != -1:
            area=(len(heights)-stack1[-2]-1)*heights[stack1[-1]]   
            ########## THIS EQUATION IS DIFFICULT TO COME UP WITH  
            max1=max(max1,area)
            stack1.pop()
            
                                    
                
        
        return max1
==========================================================================================
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.append(0)       ##appended so that everything pops out
        stack=[]
        i=0
        start=0
        maxArea=0
        # bacMax=0
        while i<=len(heights)-1:
            stack.append([heights[i],i]) 
            while len(stack)>=2 and stack[-1][0]<stack[-2][0]: ##violation condition
                height,index=stack.pop(-2)
                maxArea=max(maxArea,(i-index)*height)
                stack[-1][1]=index      ### backside is crucial     ## I update the index of this after it reverts
                
            i+=1
        return maxArea
==========================================================================================

739. Daily Temperatures-
==========================================================================================
Done, did it on my own but very tricky question!!!
------------------------------------------------------------------------------------------
1. Note how every stack solution has a while loop (not a for loop. Why because moving ahead cant be done continously)
#i pointer doesnt move ahead until it has popped all lower number behind it. 
3. This way the bottom of the stack is biggest all the time.(MONOTONIC STACKS)
  ---
 -----
--------
4. When we see a new number (at i) we pop until we hit the base which should be larger than i and keep appending answers for pop
At this point we simply add to the stack itself. Stack may become empty in this process. we simply add in that case. 

Imagine if you had multiple days in a row with a decreasing temperature, and then one very hot day - [40, 39, 38, 37, 36, 35, 34, 65]. The final day is the "answer" day for all the other days. Why? Because all the other days are in descending order (and cooler than the last day). If we make use of the fact that temperatures in descending order can share the same "answer" day, we can improve the time complexity.

In the above example, we can "delay" finding the answer for the first 7 days, and upon finding a warmer temperature 65, we can move backward to find the answer for all 7 days at the same time. This process of storing elements and then walking back through them matches the behavior of a stack.
  ---
 -----
--------
Monotonic stack diagram 
Monotonic stacks are a good option when a problem involves comparing the size of numeric elements, with their order being relevant.
------------------------------------------------------------------------------------------
Code 

class Solution(object):
    def dailyTemperatures(self, T):
        
        stack1=[]
        res=[0]*len(T)
        i=0
        while i<len(T):
            if len(stack1)>0 and T[i]>T[stack1[-1]]: ##### I used SAFROT here
                popped=stack1[-1]
                stack1.pop()
                res[popped]=i-popped
            else:
                stack1.append(i)
                i+=1
                    
        return res

============================
mine more intuitive soln 
============================
notice 2 while loops in a stack
first while loop is for greater array and 2nd is for stack operation

class Solution(object):
    def dailyTemperatures(self, temperatures):
        stack=[(temperatures[0],0)]
        ans=[0 for x in range(len(temperatures))] ###initiate with 0 because of final answer for ones which didnt pop 
        
        i=1
        while i<=len(temperatures)-1:
            while len(stack)!=0 and stack[-1][0]<temperatures[i]:  ## we pop until we find bse or exhaust stack.. remember exhaust stock clause
                a=stack.pop()
                ans[a[1]]=i-a[1]
            stack.append((temperatures[i],i))
            
            i+=1
            
        return ans
Notice the SLW pattern of while loop here , while loop for outside , 
while violation condition, 
correction 
pretend as if no violation and add to stack
-------------
ALGO_NAME: Stack_violation_correction


class Solution(object):
    def dailyTemperatures(self, temperatures):
    
        ans=[0 for j in range(len(temperatures))]
        stack=[]
        i=0
        while i<=len(temperatures)-1:
            stack.append((temperatures[i],i))           ### crucial to store the index!!
            while len(stack)>=2 and stack[-1][0]>stack[-2][0]: ##violation condition
                index=stack.pop(-2)[1]
                ans[index]=i-index  ### we keep appending answers will coming back
                
            i+=1
        return ans
==========================================================================================




==========================================================================================
155. Min Stack This is a medium level question 
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.

Logic:
You can do this by using two stacks. the second stack will keep track of mins at that point. 

class MinStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack1=[]
        self.min1=[]
        
    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack1.append(x)
        if len(self.min1)==0 or x<=self.min1[-1]:   ### EQUAL SIGN IS THE TRICK!!   Note second condition doesnt get checked when len 0. if you reverse ans is wrong 
            self.min1.append(x)               ## WHEN DO WE PUSH TO THE MIN STACK?
                                              ## if it it empty we will push or if we get a new minmum, 
                                              ## Why equal? because if there are duplicate mins in the stack, it is possible you remove one of them 
        def pop(self):
        """
        :rtype: None
        """
        if len(self.min1)>0:                     
            if self.stack1[-1]==self.min1[-1]:     ### We pop if the element getting popped is also the min element. 
                self.min1.pop()                    
        
        self.stack1.pop()

    def getMin(self):
        """
        :rtype: int
        """
        return  self.min1[-1]
==========================================================================================
716. Max Stack
------------------------------------------------------------------------------------------
This was a difficult question. 
1. Realize how to do the popMax operation. You have to go inside the stack in backward order and pop and return the first max match while popping all elements. 
2. then push these elements back in the stack. use push again to populate the maxes. In the push operation, global max variable cant be used . We only need to compare with the max at the top of the stack 
------------------------------------------------------------------------------------------
Two stacks 
-----------------
class MaxStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack1=[]
        self.max1=[]
        
    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        if len(self.stack1)==0 or x>=self.max1[-1]:
            self.max1.append(x)
        self.stack1.append(x)
        

    def pop(self):
        """
        :rtype: int
        """
        if len(self.max1)>0:                     
            if self.stack1[-1]==self.max1[-1]:     ### We pop if the element getting popped is also the min element. 
                self.max1.pop()     
        return self.stack1.pop()
        

    def top(self):
        """
        :rtype: int
        """
        return self.stack1[-1]
        
        

    def peekMax(self):
        """
        :rtype: int
        """
        return self.max1[-1]
        

    def popMax(self):
        """
        :rtype: int
        """
        b=[]
        for i in range(len(self.stack1)-1,-1,-1) :
            if self.stack1[i]==self.max1[-1]:
                ans= self.stack1.pop(i)
                self.max1.pop()             ### dont forget to pop max 
                break                       ### again dont forget to break, we dont want to pop all maxes 
            else:
                b.insert(0,self.stack1.pop(i))   ###insert in same order 

        for j in b:            
            self.push(j)
        return ans


One stack
---------------

class MaxStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack1=[]
    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        if len(self.stack1)>0:
            max1=max(x,self.stack1[-1][1])
        else:
            max1=x
        self.stack1.append([x,max1])
        

    def pop(self):
        """
        :rtype: int
        """
        return self.stack1.pop()[0]
        

    def top(self):
        """
        :rtype: int
        """
        return self.stack1[-1][0]
        
        

    def peekMax(self):
        """
        :rtype: int
        """
        return self.stack1[-1][1]
        

    def popMax(self):
        """
        :rtype: int
        """
        first=True
        b=[]
        for i in range(len(self.stack1)-1,-1,-1) :
            if self.stack1[i][0]==self.stack1[-1][1] and first==True:
                ans= self.stack1.pop(i)[0]
                first=False
            else:
                b.insert(0,self.stack1.pop(i)[0])

        for j in b:            
            self.push(j)
        return ans
==========================================================================================         
853. Car Fleet
Questions sound very tricky,but once you know it you know it. 
Calculate time needed for all cars to reach target. 
Now the relative order of cars reaching cant change. So any car which has a time lesser than a car in front is just stuck and is "merged" into the higher value
Pay attention that we float division for time. We need to sort by position!! O(nlogn)
Rest is easy once we have a regular array convert it to monotonic array by Double while violation correction method.


class Solution(object):
    logging=True
    def carFleet(self, target, position, speed):
        
        for i in range(len(position)):
            position[i]=(position[i],speed[i])
        position.sort(key=lambda x: x[0]*-1)
        
        time=[]
        for i in range(len(position)):
            time.append(float(target-position[i][0])/position[i][1] )   ##integer division
        
        ##Now time to convert a regular array to a monotonic stack big to small 
        ## didnt use extra stack here
        i=0
        count=0
        while i<=len(time)-2:
            while i>=0 and i+1<=len(time)-1 and time[i]>=time[i+1]:        ##violation
                time.pop(i+1)   
            i+=1
        #print(time)
        return len(time) 
---------------
LET THE VIOLATION HAPPEN AND THEN CORRECT IT. This is my standard Stack_violation_correction algo 

Stack based solution, where i use double while and LET THE VIOLATION HAPPEN AND THEN CORRECT IT
class Solution(object):
    logging=True
    def carFleet(self, target, position, speed):
        for i in range(len(position)):
            position[i]=(position[i],speed[i])
        position.sort(key=lambda x: x[0]*-1)
        
        time=[]
        for i in range(len(position)):
            time.append(float(target-position[i][0])/position[i][1] )   ##integer division
        ##Now time to convert a regular array to a monotonic stack big to small 
        #print(time)
        stack=[]
        i=0
        count=0
        while i<=len(time)-1:
            stack.append(time[i]) ### letting violation happen
            while len(stack)>=2 and stack[-2]>=stack[-1]:        ##violation ##len(stack)>=2 TRICKY PART
                stack.pop()
            i+=1
        #print(stack)
        return len(stack)       
==========================================================================================         
394. Decode String
How do we realize we have to use a stack?
s = "3[a2[c]]" ### This can be solved by regular processing.
Whats my logic. I try to use the Stack_violation_correction algo. I let the violation happen, then I take steps to fix it. Whats the violation here? once you find a ] thats a violation. 
Whats the correction? Unnest it. Keeping popping backwards to find string, then again to find number. Then create new string and then add it back to stack. Pretend nothing happened.
Now this the loop structure didnt exactly fit my Stack_violation_correction structure but the SPIRIT WAS THE SAME

Very clean code based on this

Not exact but inspired from and thinking process same as Stack_violation_correction
class Solution(object):
    def decodeString(self, s):
        ans=""
        stack=[]
        i=0
        while i<=len(s)-1:
            stack.append(s[i])
            if  stack[-1]=="]":
                str1=""
                stack.pop() ##remove ]
                while stack and stack[-1]!="[":
                    str1+=stack.pop() ##inverted
                stack.pop() ##remove [
                number=""
                while stack and stack[-1] in "0123456789":
                    number+=stack.pop() ##inverted
                add_string=int(number[::-1])*str1[::-1]
                for j in range(len(add_string)):
                    stack.append(add_string[j])
            i+=1
        
        return "".join(stack)  







++++++++++++++++++++++++++++++++
+++++++Group : BINARY SEARCH +++
++++++++++++++++++++++++++++++++

Q. How to deal with duplicates in binary search ?

704. Binary Search
ALGO_NAME: BINARY_SEARCH_STANDARD
#Standard binary search which we use everywhere rattofy
class Solution(object):
    def search(self, nums, target):
        
        left =0   
        right=len(nums)-1                     #initialize left and right to start and end of array 
        
        while  left<right-1: ##### THIS IS THE BASE CASE HERE AND THE MOST IMPORTANT LINE OF CODE!! IN BASE CASE I HAVE TWO NUMBERR AND LEFT AND RIGHT ARE ADJACENT!!
            mid=left+(right-left)/2   ### even it is before the mid and odd it is exactly at the mid
        	                          ### Remember this ## It will prevent integer overflow.  
            if nums[mid]==target:    #use return if target found
            	#### we will never hit an edge case because
                return mid
            elif nums[mid]>target:                            
                right=mid           ### I can keep them at the same place
            else:                   ## MEMEORY TRICK! LESSER LEFT WILL MOVE 
                left= mid                
                
        
        if nums[left]==target:
            print("strictly descending")            # In our base case only two elements are left.
            return left                             # this isnt possible so only single element
        elif nums[right]==target:
            print("strictly ascending")             # Two elements lists
            return right        
        else:
            return -1
==========================================================================================
PRAMP CODING QUESTION

Given two sorted arrays arr1 and arr2 of passport numbers, implement a function findDuplicates that returns an array of all passport numbers that are both in arr1 and arr2. Note that the output array should be sorted in an ascending order.

Let N and M be the lengths of arr1 and arr2, respectively. Solve for two cases and analyze the time & space complexities of your solutions: M Ã¢â€°Ë† N - the array lengths are approximately the same M Ã¢â€°Â« N - arr2 is much bigger than arr1.



Case 1 (M Ã¢â€°Ë† N)
======================
We can use the fact the arrays are sorted to traverse both of them in an in-order manner at the same time. The general idea of the algorithm is to use two indices, i and j, for arr1 and arr2, respectively. Every time one of indices, letÃ¢â‚¬â„¢s say i without any loss of generality, points to a value that is smaller than the value pointed by other index, we increment i. If arr1[i] == arr2[j], we add the value to the output array and increment both indices.



Case 2 (M Ã¢â€°Â« N)
======================
When one array is substantially longer than the other, we should try to avoid traversing the longer one. Instead, we can traverse the shorter array and look up its values in the longer array by using the binary search algorithm. We explain why this approach is superior in this case to the previous one in the complexity analysis section.



def find_duplicates(arr1, arr2):
  """
  M is bigger , N is smaller
  Binary Search: Nlog(M)
  Mlog(N)
  
  Two Pointer: O(M)
  """
  m=len(arr1)
  n=len(arr2)
  
  if m<n:
    m,n=n,m
    arr1,arr2=arr2,arr1
    
  # TODO
  print(arr1)
  print(arr2)
  
  #i=0
  #j=len(m)-1
  ### DO A BETTER JOB REMEMBERING THIS FUNCTION
  def binary_search(arr1,left,right,target):
    while left<right-1:
      mid=(left+right)/2
      if arr1[mid]==target:
        return arr1[mid]
      elif arr1[mid]>target:
        right=mid
      elif arr1[mid]<target:  ##LESSER LEFT WILL MOVE 
        left=mid
    
    if arr1[left]==target:
      return arr1[left]
    elif arr1[right]==target:
      return arr1[right]
    else:
      return None 
    
  ans=[]
  for i in range(len(arr2)):
    binary_search_ans=binary_search(arr1,0,m-1,arr2[i])
    if binary_search_ans:
      ans.append(binary_search_ans)
  return ans
    
#arr1 = [1, 2, 3, 5, 6, 7]
#arr2 = [3, 6, 7, 8, 20]
#print(find_duplicates(arr1,arr2))
  
# ab@gmail.com [1,2,3]
# b@gmail.com [1,3]

#[[a@gmail.com, b@gmail.com],]

    
# 8: 35am
# 35: reading question aloud
# comes up with a good brute force approach
# time complexity O(n) n => larger of the two
# space complexity O(m) m => smaller of the two
# asks whether to go for a better solution?
# 
# 39: notices that the arrays are sorted
# comes up with an optimized approach
# space complexity is minimized O(1)
# 
# 
# Compared the two appraoches using numbers
# and then concluded that binary search is better if the lengths are considerably differnet
# 
# 49: goes into implementation mode
# 
# implements binary search as a part of the function (this could have been a separate function)
# 
# 52: converts binary search into a function (no hints given)
# 56: variable names could be better - smaller / larger etc.
# 
# 59: during dry run asking "which is the bigger array?" - this could have been clearer using the variables
# asking leading questions
      
      
      
  
#1. Test cases 
#2. Dry run on test cases
#3. Use smaller example
#4. Asking leading questions
#5. Readability and maintaibility, comments
# 25 minutes
==========================================================================================
162. Find Peak Element
Tag : Binary search 
------------------------------------------------------------------------------------------
Done
------------------------------------------------------------------------------------------
logic 
1. can be solved by linear scan o(n)
2. can be solved by binary search 0(logn)
We use binary search here. Why ? Tricky 
When you check the middle, if the element on the left subarray is greater than the middle element we know that it either continues to grow and reaches a peak at the end of the subarray (since the bounds are -inf), or it eventually shrinks. Either way, we know that there must exist a peak on the left. The same is true for the right subarray.

Note that the end points are considered as peaks
------------------------------------------------------------------------------------------
Code:
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left =0   
        right=len(nums)-1                     #initialize left and right to start and end of array 
        
        while  left<right-1 ##### THIS IS THE BASE CASE HERE AND THE MOST IMPORTANT LINE OF CODE!! IN BASE CASE I HAVE TWO NUMBERR AND LEFT AND RIGHT ARE ADJACENT!!
            
        	mid=left+(right-left)/2   ### even it is before the mid and odd it is exactly at the mid
        	                          ### Remember this ## It will prevent integer overflow.  
           
            if nums[mid]>nums[mid+1] and nums[mid]>nums[mid-1]:    #use return if peak found
            	#### we will never hit an edge case because
                return mid
            elif nums[mid-1]>nums[mid]:                            
                right=mid           ### I can keep them at the same place
            else:
                left= mid                
                
        
        if nums[left]>nums[right]:
            print("strictly descending")# In our base case only two elements are left. Compare and return peak
            return left                 # This part is difficult to feel but verfied it
                                        # [6,5,4,3,2,3,2] ###peak at the edge
        else:
            print("strictly ascending")
            return right
###################################################################################
## when its descending the pointers move to the left end otherwise to the right end 
###################################################################################
Code 2 with no edge cases: ## see if this can be generalized to other questions, cant yet
    while left<right:
            mid= left+(right-left)/2  
            
            if nums[mid]>nums[mid+1]: ## peak found
                right=mid
            else:
                left=mid+1
        return left

search code 3 with no edge cases   ## check if can generalized ## doesnt work with peak element
        while left<=right:
            mid=left+(right-left)/2
            
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                right=mid-1
            else:
                left=mid+1

==========================================================================================
852. Peak Index in a Mountain Array
------------------------------------------------------------------------------------------
Done
------------------------------------------------------------------------------------------
Tag : Binary search. Same question but different in the way that it cant be strictly ascending or descending

##same thing as above except the strict asc/desc handling part

class Solution(object):
    def peakIndexInMountainArray(self, arr):
        left=0
        right=len(arr)-1
        
        while left<right-1:
            mid = left+(right-left)/2
            if arr[mid+1]<arr[mid] and arr[mid-1]<arr[mid]:
                return mid
            elif arr[mid+1]>arr[mid]:
                left=mid 
            elif arr[mid-1]>arr[mid]:
                right=mid
        
        if arr[left]>arr[right]:     ##this part is not needed as peak isnt at the edge
            return left
        else:
            return right
==========================================================================================   
35. Search Insert Position
Tag : Binary search.
------------------------------------------------------------------------------------------
Done
------------------------------------------------------------------------------------------
Logic: I still want to stop my binary search at the last two elements.
After stopping,
1. it can still be equal to left or right
2. It can be smaller than left, (only possible when left is at 0) ## this happens when we get a target lower than min 
3. It can be larger than right, (only possible when right is at max length) ## this happens when we get a target lower than min 
4. target can be in mid of right and left 

class Solution(object):
    def searchInsert(self, nums, target):
        left=0
        right=len(nums)-1
        while left<right-1:
            mid=left+(right-left)/2
            
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                right=mid
            else:
                left=mid
        ### edge cases
        if target<nums[left] or nums[left]==target:   
            return left
        elif nums[right]==target or (target<nums[right] and target>nums[left]): 
            return right
        elif target>nums[right]:   
            return right+1 
==========================================================================================
153. Find Minimum in Rotated Sorted Array
1st part of next question and this is the same -- see explanation there

We just need to find the pivot index. What is the pivot ?
[2,3,4,0,1] ## (index)3 is the pivot here. Every element's index is pushed by pivot. 

class Solution(object):
    def findMin(self, nums):
        left=0
        right=len(nums)-1
        ###################### finding the pivot 
        while left<right-1:
            mid=left+(right-left)/2
            
            if nums[mid]<nums[mid-1]:   ### the array only increases and drops only at one point
                pivot=mid
                break
            elif nums[mid]>nums[right]:    #### This condition is tricky and weird, draw the graph!!! 
                left=mid                   #### Check bottom for better explanation
            else:
                right=mid
        else:
            ### the previous loop didnt break so this is running now to find pivot
            if nums[left]>nums[right]:
                pivot=right           ### pivot by definition is smaller, see the diagram
                                      ### This happens only in case of stricltly descending. In this case, SD can only happen 
                                      ### if len=2
            else:
                pivot=0               ####case of sorted array ## when strictly ascending pointers move to the right end
                                      ####We know and remember this
                    
        return nums[pivot]
==========================================================================================
153. Find Minimum in Rotated Sorted Array II
Dups is the main issue

extra case: Case 3). nums[pivot] == nums[right]
In this case, we are not sure which side of the pivot that the desired minimum element would reside.
To further reduce the search scope, a safe measure would be to reduce the upper bound by one (i.e. high = high - 1), rather than moving aggressively to the pivot point.

class Solution(object):
    def findMin(self, nums):
        left=0
        right=len(nums)-1
        ###################### finding the pivot 
        while left<right-1:
            mid=left+(right-left)/2
            
            if nums[mid]<nums[mid-1]:   ### the array only increases and drops only at one point
                pivot=mid
                break
            elif nums[mid]>nums[right]:    #### This condition is tricky and weird, draw the graph!!! 
                left=mid                   #### Check bottom for better explanation
            elif nums[mid]<nums[right]: 
                right=mid
            elif nums[mid]==nums[right]:
                #if right != 0 and nums[right] >= nums[right-1]:
                right-=1
                
        else:
            ### the previous loop didnt break so this is running now to find pivot
            if nums[left]>nums[right]:
                pivot=right           ### pivot by definition is smaller, see the diagram
                                      ### This happens only in case of stricltly descending. In this case, SD can only happen 
                                      ### if len=2
            else:
                pivot=0               ####case of sorted array ## when strictly ascending pointers move to the right end
                                      ####We know and remember this
                    
        return nums[pivot]





==========================================================================================
33. Search in Rotated Sorted Array
------------------------------------------------------------------------------------------
Done very difficult
------------------------------------------------------------------------------------------
What is the pivot ?
Logic: 1. binary search 1 to find the pivot. what should be condn?
       2. binary search again to find target. what should be condn?
------------------------------------------------------------------------------------------

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        
        if len(nums)==0:
            return -1

        left=0
        right=len(nums)-1
        ###################### finding the pivot 
        while left<right-1:
            mid=left+(right-left)/2
            
            if nums[mid]<nums[mid-1]:   ### the array only increases and drops only at one point
                pivot=mid
                break
            elif nums[mid]>nums[right]:    #### This condition is tricky and weird, draw the graph!!! 
                left=mid                   #### Check bottom for better explanation
            else:
                right=mid
        else:
            ### the previous loop didnt break so this is running now to find pivot
            if nums[left]>nums[right]:
                pivot=right           ### pivot by definition is smaller, see the diagram
            else:
                pivot=0               ####case of sorted array ## when strictly ascending pointers move to the right end
                                      ####We know and remember this
    
        left=0
        right=len(nums)-1
        
        if target>nums[right]:     #### Again we check if our target is to the right to left of vertical line 
                                   #### by comparing with height of R and search only in the corresponding wing
            right=pivot-1
        else:
            left=pivot
        
        ###################### standard search again
        
        while left<right-1:
            mid=left+(right-left)/2
            
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                right=mid
            else:
                left=mid
        if nums[left]==target:
            return left
        elif nums[right]==target:
            return right
        return -1
        
####We just need to find out where the mid is to the right or left of vertical line. 
####So we compare to R height. if its taller than R, its on the right side. We want answer to be in between L and R so we move left to mid, else move right
    pivot-1 
      -  
    - |
  -   |   
-     |     -
L     |    -R
      |  -
      |-    
    pivot
==========================================================================================
69. sqrt(x) 
Tag : Binary search
-------------------------
Done
-------------------------
1. Remove edge cases 0, 1, 2 etc 
2. In my binary search when left and right are adjacent things stop. So decimal issue resolved by simply taking left 
3. after 4 , sqrt > n/2, so we can choose n/2+1 as right
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x<2:       ##edge cases
            return x
        left=0
        right=x/2+1 ##anything bigger than x/2 is allowed, we dont want x/2 because 3,4,5 will have right as answer
        while left<right-1:
            mid = left+(right-left)/2
            if mid*mid==x :
                return mid
            elif mid*mid>x:
                right=mid 
            else:
                left=mid
        return left

74. Search a 2D Matrix
Tag: Matrix, Binary Search
------------------------------------------------------------------------------------------
## Done, 
It is still a binary search just convert the mid index to corresponding index in the matrix. 
How /n and %n
------------------------------------------------------------------------------------------
class Solution(object):
    def searchMatrix(self, matrix, target):
        m=len(matrix)
        n=len(matrix[0])
        
        left=0
        right=m*n-1
        
        while left<right-1:
            mid=left+(right-left)/2
            if matrix[mid/n][mid%n]==target:
                return True
            elif matrix[mid/n][mid%n]>target:
                right=mid
            else:
                left=mid
        if matrix[left/n][left%n]==target or  matrix[right/n][right%n]==target:
            return True
        return False
====================================================================== 
278. First Bad Version
## Done,  standard binary search with the end case important 
class Solution(object):
    def firstBadVersion(self, n):

        left=1
        right=n
        
        while left<right-1:
            mid=left+(right-left)/2
            if isBadVersion(mid) and not isBadVersion(mid-1):
                return mid
            elif isBadVersion(mid):
                right=mid
            else:
                left=mid
        
        if isBadVersion(left):
            return left
        elif isBadVersion(right):
            return right
======================================================================       
441. Arranging Coins
We basically need to find the number of rows. it can be max N (N+1) to include edge case of 1
if we have mid 


# simple while loop and subtract for each level
class Solution(object):
    def arrangeCoins(self, n):
        i=1
        while n>=i:
            n-=i
            i+=1
        return i-1
------------------------------------------------------------------------------------------
##Simple sum check 
for x in range(1,n+1,1):
    if n<x*(x+1)/2:
        return x-1
    elif n==x*(x+1)/2:
        return x
------------------------------------------------------------------------------------------
##Optimize sum check using binary search ### exactly like search insert position 
class Solution(object):
    def arrangeCoins(self, n):
        """
        :type n: int
        :rtype: int
        """
        left=1
        right=n+1  ## case of 1 
        
        while left<right-1:
            mid=left+(right-left)/2
            if mid*(mid+1)/2==n:
                return mid
            elif mid*(mid+1)/2>n:
                right=mid
            else:
                left=mid
                
        if left*(left+1)/2==n:
            return left
        elif right*(right+1)/2==n:
            return right
        elif n<left*(left+1)/2 (not possible) or (n>left*(left+1)/2 and n<right*(right+1)/2):
            return left
        else:
            return right
======================================================================       
367. Valid Perfect Square
## Done, 
##blindly implementing binary search logic ## very similar to sqrtx
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        left=0
        right=num/2+1 ###+1 is important
        
        while left<right-1:
            mid=left+(right-left)/2
            if mid*mid==num:
                return True
            elif mid*mid>num:
                right=mid
            else:
                left=mid
        
        if left*left==num or right*right==num:
            return True
        else:
            return False
======================================================================   
# k will lie between 1 to max element, because There is no point of k > max element
# Can k be less than min element? yes if H is very high    
# so binary search between 1 and max
## given a k can we eat all?
## we need to divide each number by k. then add the hrs. if remainder==0 (quotinent) else quotient+1 . if sum<=h then we can 
875. Koko Eating Bananas
Done, did this on my own with some iterations
class Solution(object):
    def minEatingSpeed(self, piles, h): 
        ## piles.sort()  ## sorting on piles isnt needed as binary search is happening on k not here
        left=1
        right=max(piles)
        while  left<right-1: 
            mid=left+(right-left)/2   
            noHours= sum([x/mid if x%mid==0 else (x/mid)+1 for x in piles])
            #print(mid,noHours)
            if noHours<=h:          ## even if its equal try to go lower !!!!                         
                right=mid           
            else:
                left= mid                
        ## always reduce to base case ad then check
        if sum([x/left if x%left==0 else (x/left)+1 for x in piles])<=h:       ## base case 2 remaining
            return left  
        elif sum([x/right if x%right==0 else (x/right)+1 for x in piles])<=h:          
            return right         
======================================================================   
981. Time Based Key-Value Store
# All the timestamps timestamp of set are strictly increasing.
class TimeMap(object):

    def __init__(self):
        self.dict1={}
        
    def set(self, key, value, timestamp):
        if key in self.dict1:
            self.dict1[key].append([value,timestamp])
        else:
            self.dict1[key]=[[value,timestamp]]
              

    def get(self, key, timestamp):
        if key in self.dict1:
            list1=self.dict1[key]
            ## [[v1,1],[v2,2],[v3,3]]
            #print(self.dict1)
            for i in range(len(list1)-1,-1,-1):
                if timestamp>=list1[i][1]:
                    return list1[i][0]
            return ""       ## timestamp is too less
        return ""           ## key is not present
------------------------------------------------------------------------------------------
def get(self, key, timestamp):
        if key in self.dict1:
            list1=self.dict1[key]
            ## [[v1,1],[v2,2],[v3,3]]
            ##print(self.dict1)
            left=0
            right=len(list1)-1

            while  left<right-1: 
                mid=left+(right-left)/2   
                if list1[mid][1]>timestamp:                                 
                    right=mid           
                elif list1[mid][1]<timestamp: 
                    left= mid 
                else:
                    return list1[mid][0]
                    
            if list1[right][1]<=timestamp:
                return list1[right][0]
            elif list1[left][1]<=timestamp:
                return list1[left][0]
            else:
                return ""
        else:
            return ""
======================================================================   

======================================================================       
Pancake sorting 
-----------------------------------------------------------------------
Logic: Each time we need to find max (find its index) and then flip 
-----------------------------------------------------------------------

class Solution(object):
    def pancakeSort(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        if A==sorted(A):
            return []
        
        ret=[]

        for j in range(len(A)-1,-1,-1):
            index=self.findMax(A[:j+1])
            A[:index+1]=A[:index+1][::-1]
            #print(A)
            ret.append(index+1)
            A[:j+1]=A[:j+1][::-1]
            #print(A)
            ret.append(j+1)
        
        return ret 


    def findMax(self,nums):
        maxi=0
        max1= float("-inf")
        for i in range(len(nums)):
            if nums[i]>max1:
                max1=nums[i]
                maxi=i
        return maxi
        
======================================================================


++++++++++++++++++++++++++++++++++++++
+++++++Group : DP +++
++++++++++++++++++++++++++++++++++++++
My DP rules
========
i indicates array from i going till end!!! Remember ALWAYS DO THIS





https://leetcode.com/problems/target-sum/discuss/455024/DP-IS-EASY!-5-Steps-to-Think-Through-DP-Questions.

https://stackoverflow.com/questions/12133754/whats-the-difference-between-recursion-memoization-dynamic-programming
https://stackoverflow.com/questions/6164629/what-is-the-difference-between-bottom-up-and-top-down
https://www.geeksforgeeks.org/tabulation-vs-memoization/
-----------------------------------------------------------------------
fib_cache = {}
###### TOP DOWN DP USING MEMOIZATON 
def memo_fib(n):
  global fib_cache
  if n == 0 or n == 1:
     return 1
  if n in fib_cache:
     return fib_cache[n]
  ret = memo_fib(n - 1) + memo_fib(n - 2)
  fib_cache[n] = ret
  return ret


###### BOTTOM UP DP USING Tabulation
def dp_fib(n):
   partial_answers = [1, 1]
   while len(partial_answers) <= n:
     partial_answers.append(partial_answers[-1] + partial_answers[-2])
   return partial_answers[n]

print memo_fib(5), dp_fib(5)
-----------------------------------------------------------------------






top-down DP is basically (divide and conquer)+ memoization.
Divide&Conquer can be done using recursion or while loop 

Bottom-Up Approach uses Tabulation -- An act of creating a "table"

509. Fibonacci Number
Tag: DP
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
#Simple recursion without memoization/tabulation
Head recursion because answer not found until the head
class Solution:
    def fib(self, N):
        if N <= 1:
            return N
        return self.fib(N - 1) + self.fib(N - 2)
Time complexity: O(2^n) 
Space complexity: O(N) and NOT O(2^n) 
https://www.youtube.com/watch?v=P8Xa2BitN3I
------------------------------------------------------------------------------------------
Bottom-Up Approach using Tabulation
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0:
            return 0
    
        f=[0,1]
        
        for i in range(1,n,1):
            f.append(f[i]+f[i-1])
        return f[-1]
Time complexity: O(N) 
Space complexity: O(N)
----------------------------------------------------------------------
Bottom-Up Approach using Tabulation can be further simplified to Space O(1) as we realize we only use 2 last ones 
and dont need to store the entire table
Time complexity: O(N) 
Space complexity: O(1)
----------------------------------------------------------------------
Top-Down Approach using Memoization
class Solution:
    cache = {0: 0, 1: 1}

    def fib(self,N):
        if N in self.cache:
            return self.cache[N]
        self.cache[N] = self.fib(N - 1) + self.fib(N - 2)
        return self.cache[N]
Time complexity: O(N) 
Why ? https://www.youtube.com/watch?v=P8Xa2BitN3I
Space complexity: O(N)
======================================================================
70. Climbing Stairs
# f[i] is the number of distinct ways you can reach the ith step
# f[i]= f[i-1]+ f[i-2]
# f[3]=f[1]+ f[2]
#  3   = 1 + 2
----------------------------------------------------------------------
Simple Recursion
class Solution(object):
    def climbStairs(self, n):
        if n==2:
            return 2     ####base cases
        if n==1:
            return 1     #### base cases
        return self.climbStairs(n-1)+self.climbStairs(n-2)
----------------------------------------------------------------------
Top down with memo 
class Solution(object): 
    def climbStairsR(self,i):
        if i==2:
            return 2     ####base cases
        if i==1:
            return 1     #### base cases
        if i in self.memo:
            return self.memo[i]
        
        a=self.climbStairsR(i-1)
        b=self.climbStairsR(i-2)
        self.memo[i]=a+b
        return self.memo[i]  
    def climbStairs(self, n):
        self.memo={}
        return self.climbStairsR(n)
----------------------------------------------------------------------
##Bottom up tabulation 
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        f=[1,2]
        for i in range(2,n,1):
            f.append(f[i-1]+f[i-2])
        return f[n-1]
----------------------------------------------------------------------

======================================================================
746. Min Cost Climbing Stairs

This question can also be done by graph traversal, more expensive ofcourse.
## f(i)=min(f[i-1]+cost[i-1],f[i-2]+cost[i-2])
## f(i) is the min cost to reach i staircase, so it shouldnt be inclusive of cost of stair itself
## we need to find f(n)
----------------------------------------------------------------------
##Bottom up tabulation 
class Solution(object):
    def minCostClimbingStairs(self, cost):
        n=len(cost)
        f=[0,0]
        for i in range(2,n+1):
            f.append(min(f[i-1]+cost[i-1],f[i-2]+cost[i-2]))
        return f[n]
O(N)- space complexity
----------------------------------------------------------------------
##Bottom up tabulation can be optimized by realizing we only need two previous elements
class Solution(object):
    def minCostClimbingStairs(self, cost):
        n=len(cost)
        f=[0,0]
        for i in range(2,n+1):
            a=f[1]
            f[1]=min(f[1]+cost[i-1],f[0]+cost[i-2])
            f[0]=a
        return f[-1]
O(1) Space
----------------------------------------------------------------------
### TOP DOWN -- MEMOIZATION 
## EXACTLY LIKE HEAD RECURSION EXCEPT YOU ADD THINGS TO DICT
1. Base case
2. Memo check condition and return if found 
3. assume sub problems are known like in HEAD recursion and use that to create answer and put it in memo and then return that memo. 


class Solution(object):
    def minimum_cost(self,i,cost):
            
        if i <= 1:  # Base case, we are allowed to start at either step 0 or step 1
            return 0

        # Check if we have already calculated minimum_cost(i)
        if i in self.memo:
            return self.memo[i]

        # If not, cache the result in our hash map and return it
        down_one = cost[i - 1] + self.minimum_cost(i - 1,cost) 
        down_two = cost[i - 2] + self.minimum_cost(i - 2,cost)
        self.memo[i] = min(down_one, down_two)
        return self.memo[i]
        
    def minCostClimbingStairs(self, cost):
        self.memo = {}
        return self.minimum_cost(len(cost),cost)
======================================================================
279. Perfect Squares
Tags: DP 
-------------------------------------------------------------------------------------------------------------------------
Logic:
Method0: Recursion with no memo/tabulation (Not done)
Method1: bottom up DP with tabulation
A given number can be made by all the perfect squares below it. So consider a dp where ith index of the dp answr of n=i
Now, dp[12]= min(dp[12-1*1]+1,dp[12-2*2]+1,dp[12-3*3])+1 ### j*j can move till the j*j<=1
Answer is correct but it is exceeding time limit. I was able to come up with this on my own
Method2: Simple BFS

Group: Coin change (exact same almost)
-------------------------------------------------------------------------------------------------------------------------
### Bottom-Up Approach using Tabulation - TLE
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        f=[0,1]
        for i in range(2,n+1,1):
            f.append(float('inf'))
            for j in range(i):
                if j*j>i:
                    break
                f[i]=min(f[i],f[i-j*j]+1)
        
        return f[n]
-------------------------------------------------------------------------------------------------------------------------
# Bottom-Up Approach using Tabulation but instead of calculating square terms each time we store them -- Accepted
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        square_nums = [i**2 for i in range(1,n)]
        list1=[0,1]
        for i in range(2,n+1,1):
            minM=float("inf")
            for square in square_nums:
                if square>i:
                    break
                else:
                    minM=min(minM,list1[i-square]+1)
            list1.append(minM)
        
        return list1[-1]
-------------------------------------------------------------------------------------------------------------------------
There is some waste in this. Why ? When we reach square numbers we still do more calculations. But hey these are square numbers and answer is 1. No need to check.
Time complexity: O(n*root(n)). Why ?
We fill an dp array of len n but at each step we check at max root(i) terms 
So its root(1)+root(2)+root(3)+... n terms
Space complexity: O(n)
-------------------------------------------------------------------------------------------------------------------------
Why BFS works here? We are doing a level by level traversal and stopping when we find a perfect square. This works because 
the count of levels itself is the answer. There is no other calculation.


# Top down Approach -- used BFS to traverse tree (didnt use memoization) -- TLE
class Solution(object):
    def numSquares(self, n):
        square_nums = [i**2 for i in range(1,int(n**0.5)+1)]
        queue=[n]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                neighbors=[current-square_num for square_num in square_nums]
                for neighbor in neighbors:
                    queue.insert(0,neighbor)
                if current in square_nums: ##### FOR THE LOVE OF GOD PLEASE CHECK AND RETURN LEVELS HERE
                    return level+1  ## we want 1,2,3 levels so I added 1 if we wanted 0,1,2 it would be level
            level+=1 ##### FOR THE LOVE OF GOD PLEASE INCREASE LEVELS HERE
-----------------------------------------------------------------------------------------------------------
Somehow this is faster than previous BFS
# BFS with one level opti --still TLE 
class Solution(object):
    def numSquares(self, n):
        square_nums = {i*i:1 for i in range(1,int(n**0.5)+1)}
        queue=[n]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                if current in square_nums:
                    return level+1  ## we want 1,2,3 levels so I added 1 ##since we are checking before insert 
                neighbors=[current-square_num for square_num in square_nums.keys()]
                for neighbor in neighbors:
                    if neighbor in square_nums:
                        return level+2  ## checking before insert will L+2 as we want 1,2,3(+1) and its inside(+1)
                    queue.insert(0,neighbor)
                
            level+=1
-----------------------------------------------------------------------------------------------------------
# BFS with one level opti and visited opti 
class Solution(object):
    def numSquares(self, n):
        square_nums = {i*i:1 for i in range(1,int(n**0.5)+1)}
        queue=[n]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                if current in square_nums:
                    return level+1  ## we want 1,2,3 levels so I added 1 ##since we are checking before insert 
                neighbors=[current-square_num for square_num in square_nums.keys()]
                for neighbor in neighbors:
                    if neighbor in square_nums:
                        return level+2  ## checking before insert will L+2 as we want 1,2,3(+1) and its inside(+1)
                    queue.insert(0,neighbor)
                
            level+=1
------------------------------------------------------------------------------------------
Top down approach with memo -- TLE dont know why 

class Solution(object):
    def numSquaresR(self, i):
        square_nums = [x**2 for x in range(1,int(i**0.5)+1)]
        if i == 1:
            return self.memo[1]
        if i in square_nums:
            return 1
        
        if i in self.memo:
            return self.memo[i]
        
        minM=float("inf")
        for j in range(len(square_nums)):
            if i>square_nums[j]:
                minM=min(minM,self.numSquaresR(i-square_nums[j])+1)
            else:
                break
        self.memo[i]=minM
        return self.memo[i]
    
    def numSquares(self, n):
        self.memo={1:1}
        return self.numSquaresR(n)



======================================================================
            
 322. Coin Change
https://leetcode.com/problems/coin-change/discuss/1320117/Python-2-approaches-%3A-BFS-Top-down-Memoized-recursion-%3A-Explained-%2B-visualized

 Tags: DP, UNBOUNDED KNAPSACK
------------------------------------------------------------------------------------------
tricky question due to edge cases 
Logic: question is same as the question Pefect Squares
We just need to reduce the amount by all possible amounts and check the minimum
1. Bottom-Up Approach using Tabulation -- easiest
2. Simple BFS - TLE
3. BFS with visited
4. Memoized recursion -- later  
------------------------------------------------------------------------------------------
#1. Bottom-Up Approach using Tabulation (1D ARRAY)
What is the TREE STRUCTURE WE USE HERE?  TREE STRUCTURE: CHOOSE_ALL_POSSIBLE_CHILDREN_WITHOUT_INDEX_FORWARDING 
11,[1,2,5]
10,     9,      6      ### Children  

So at each step we keep reducing the amount by all possible coins which are shorter that the root.
CHOOSE_ALL_POSSIBLE_CHILDREN_WITHOUT_INDEX_FORWARDING WORKS FOR UNBOUNDED KNAPSACK,because repitition is allowed. 
Even while repition is allowed this method gives rise to permutations and doesnt create unique sets. 


Here, what is the dp f?
f is the fewest number of coins that you need to make up that amount i


class Solution(object):
    def coinChange(self, coins, amount):
        coins.sort()     ### this is important since I am breaking 
        
        f=[0]            ### Because we need 0 coins to make up amount 0
        
        for i in range(1,amount+1,1):
            f.append(float("inf"))  ### since I am doing a min operation later
            for j in range(len(coins)):
                if coins[j]>i:      ### I use all the smaller coin that our target i
                    break
                f[i]= min(f[i],f[i-coins[j]]+1)
            
        
        if  f[-1]==float("inf"):
            return -1
        else:
            return f[-1]  

Minor edit can also be done without sorting but checking all 
        if coins[j]<=i:
            f[i]= min(f[i],f[i-coins[j]]+1)
------------------------------------------------------------------------------------------
THEORY TIME
APPARENTLY COIN CHANGE IS A SUBCATEGORY OF PROBLEMS CALLED UNBOUNDED KNAPSACK.
what is identifying nature of this type ?
You have a bag and a target, for each item in the bag you can any number of times to get to the target.
We can choose only once in 0/1 KNAPSACK

TREE STRUCTURE: TAKE_OR_LEAVE_BUT_UNBOUNDED

This TREE STRUCTURE IS KINDA TRICKY TO THINK ABOUT, we take or leave the index. binary choice. 
BUTTTTT  WHILE LEAVING ELEMENT INDEX MOVES FORWARD, WHILE TAKING IT DOESNT BECAUSE WE WANT TO RETAKE THE SAME ELEMENT.
Questions using this structure: Combination sum, Coin change 





Now in the BOTTOM_UP_TABULATION approach, you should have 2D grid of (1 to target) in the columns and one row for every element in the KNAPSACK!!
You have to reach the bottom right corner of the grid which is your answer.

what is f: ## f[i][j] is the minimum coins to reach target j using coins starting from index (start till i)
Going backwards
f[i][j] is the minimum coins to reach target j using coins starting from index (i till end)

Mistake: Treated this like 0/1 knapsack and tried to make 0/1 selection at 1st index. but the TREE Structure is 
TREE STRUCTURE: TAKE_OR_LEAVE_BUT_UNBOUNDED. After making the 1 choice we cant move ahead in index!!! 

# i-> 0 to n 
# k-> 0 to target
# Base cases:
# i=n, no more decisions to make 
# f(n,k) is 1 when k=0 else infinite (can never make these amounts)
Watch this video: Very crucial. Matrix operations are very difficult. Need some element of memory.
https://www.youtube.com/watch?v=ZI17bgz07EE

#1. Bottom-Up Approach using Tabulation (2D ARRAY)

ALGO_NAME:BOTTOM_UP_TABULATION_2D_BACKWARDS 
LOGIC POINTS
1. Range for rows is index while column is in actual column range. This is based on the recurrence relationship which uses index for i but actual value for column. 
2. Fill the base cases first 
3. Decide what is the sequence of calculation, 
SYNTAX POINTS
1. When we are in dp[i][k]. No conversion is needed for i because we use index directly for i.
   But conversion is needed for k. 
2. When I refer the actuals array of rows,coins here, conversion is needed again. 




GOING FORWARD WHICH I DONT USE
https://leetcode.com/problems/coin-change/discuss/720880/Python-DP-Explanation-with-Pictures
class Solution(object):
    logging=False
    def coinChange(self, coins, amount):
        n = len(coins)
        
        dp = [[None for k in range(amount + 1)] for i in range(n + 1)]
        #### Filled the 0th row
        for k in range(amount + 1): ### Use actual column range again 
            dp[0][k] = float("inf") ### No need for conversion of k to index. But do if necessary
                                    ### 0th row doesnt include 0th element
        
        #### Filled the 0th column 
        for i in range(n + 1):
            dp[i][0] = 0
        if self.logging: print(dp)
        
        #### Now for rest of the numbers, I just make sure index is within bounds 
        #### and I consider two options, copying from the column above dp[i-1][j]
        #### and subtracting the amount by coin value and stay in the same row 
        for i in range(1,n + 1):
            for j in range(1,amount + 1):

                if j -coins[i]>=0:
                    take = 1 + dp[i][j - coins[i]]
                    leave = dp[i - 1][j]
                    dp[i][j] = min(take, leave)
                    
                else:
                    dp[i][j] = dp[i - 1][j]

        if  dp[-1][-1]==float("inf"):
            return -1 
        else:
            return dp[-1][-1]
------------------------------------------------------------------------------------------
ALGO_NAME:BOTTOM_UP_TABULATION_2D_BACKWARDS : USE THIS

class Solution(object):
    logging=False
    def coinChange(self, coins, amount):
        n = len(coins)
        
        dp = [[None for k in range(amount + 1)] for i in range(n + 1)]
        #### Filled the nth row
        for k in range(amount + 1): ### Use actual column range again 
            dp[n][k] = float("inf") ### No need for conversion of k to index. But do if necessary
                                    ### nth row has 0 elements so infinite number of coins are needed 
                                    ### to create any sum 
        
        #### Filled the 0th column 
        for i in range(n + 1):      ###the min coins to create amount 0 is 0
            dp[i][0] = 0
        if self.logging: print(dp)
        
        #### Now for rest of the numbers, I just make sure index is within bounds 
        #### and I consider two options, copying from the column above dp[i-1][j]
        #### and subtracting the amount by coin value and stay in the same row 
        for i in range(n-1,-1,-1):   
            for k in range(1,amount + 1):

                if k -coins[i]>=0:  ###
                    ####this recurrence is very unintuitve to write
                    #### how is the tree formed here, see 
                    #### TREE STRUCTURE: TAKE_OR_LEAVE_BUT_UNBOUNDED
                    dp[i][k] = min(1 + dp[i][k - coins[i]], dp[i + 1][k] )
                    
                else:
                    dp[i][k] = dp[i + 1][k]

        if  dp[0][amount]==float("inf"):
            return -1 
        else:
            return dp[0][amount]

------------------------------------------------------------------------------------------
TREE STRUCTURE: CHOOSE_ALL_POSSIBLE_CHILDREN_WITHOUT_INDEX_FORWARDING 
#2a).Simple BFS like the previous question TLE
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        queue=[amount]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                neighbors=[current-coin for coin in coins]
                for neighbor in neighbors:
                    if neighbor>=0:             ### important because loop wont end if you dont put this
                        queue.insert(0,neighbor)
                if current==0:
                    return level   ###0,1,2 return is there 
                
            level+=1
        return -1
------------------------------------------------------------------------------------------
#2b) BFS with one level neighbor optimization  -- TLE
class Solution(object):
    def coinChange(self, coins, amount):
        queue=[amount]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                if current==0:                      ###one level neighbor opti
                    return level
                neighbors=[current-coin for coin in coins]
                for neighbor in neighbors:
                    if neighbor>=0:            
                        if neighbor==0:             ###one level neighbor opti
                            return level+1
                        queue.insert(0,neighbor)
            level+=1
        return -1
------------------------------------------------------------------------------------------
BFS with visited is better. Why ? If the same node repeats at lower levels, it might give us anser but its not better than the answer we can get from using the node at a higher level. so we remove that branch.
still TLE
#2c) BFS with visited optimization -- TLE 

class Solution(object):
    def coinChange(self, coins, amount):
        queue=[amount]
        level=0
        visited={}
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                neighbors=[current-coin for coin in coins]
                for neighbor in neighbors:
                    if neighbor>=0 and neighbor not in visited:
                        queue.insert(0,neighbor)
                if current==0:
                    return level   ###0,1,2 return is there 
                visited[current]=1
            level+=1
        return -1
------------------------------------------------------------------------------------------
#2d) BFS with visited optimization and one level neighbor optimization -- Accepted
class Solution(object):
    def coinChange(self, coins, amount):
        queue=[amount]
        level=0
        visited={}
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                if current==0:
                    return level
                neighbors=[current-coin for coin in coins]
                for neighbor in neighbors:
                    if neighbor>=0 and neighbor not in visited:            
                        if neighbor==0:     ###one level neighbor opti
                            return level+1
                        visited[neighbor]=1 ###visited  opti needs TO BE INSIDE!!!!!
                        queue.insert(0,neighbor)        
            level+=1
        return -1
#######################################################################
#visited opti is the better optimization, but it needs to be done before insertion to prevent dups in queue!!!! 
#######################################################################
------------------------------------------------------------------------------------------


==========================================================================================
518. Coin Change 2
Coin change I -- asks for the fewest number of coins which can give us the target, so we grow the tree using TREE STRUCTURE: CHOOSE_ALL_POSSIBLE_CHILDREN and we KNOW that there are duplicates but we DONT CARE! 
Now in this question too we need to grow the tree but now we want DISTINCT combinations. 

TRICK!!!!
So for DEDUPLICATING while TREE GROWTH I use a STANDARD TECHNIQUE. I PASS THE coins_rem ARRAY as a state variable.
At each coin addition, the below tree can only get the coins including and after that index. Used the same trick in all subset problems. 
The other option was doing the same thing but with index. Havent tried that.
THIS TREE STRUCTURE IS (TREE STRUCTURE: CHOOSE_ALL_POSSIBLE_CHILDREN_WITH_INDEX_FORWARDING)
This tree structure ensures that there are no duplicates and works with UNBOUNDED
IT WILL ALSO WORK WITH BOUNDED, WE JUST NEED TO MOVE THE INDEX 1 more I think.


## ALGO_NAME: TREE_STYLE_DFS_HEAD_RECURSIVE -- TLE
## duplicate removal will be an issue
## I added coins_rem for that so that we only take indexes greater than that 
class Solution(object):
    logging=False
    def dfsR(self,root,amount,coins):
        coins_rem,pathSum=root
        ## base case
        if pathSum>amount:      ## DONT RETURN 1 if it exceeds LOL
            return 0
        if pathSum==amount:
            #if self.logging: print("leaf node",pathSum)
            return 1
        
        if root in self.memo:
            return self.memo[root]
        
        ##create neighbors
        neighbors=[]
        for i in range(len(coins_rem)):
            neighbors.append((coins_rem[i:],pathSum+coins_rem[i]))
        if self.logging: print("current",pathSum)
        if self.logging: print("neighbors",neighbors)
        
        rootList=[]
        for neighbor in neighbors:
            neighbor_ans=self.dfsR(neighbor,amount,coins)
            rootList.append(neighbor_ans)
        
        self.memo[root]=sum(rootList)
        return self.memo[root]
            
    
    def change(self, amount, coins):
        self.memo={}
        return self.dfsR((tuple(coins),0),amount,coins)
----------------------------------------------------------------------------------------
ALGO_NAME:BOTTOM_UP_TABULATION_2D_BACKWARDS 
THE TREE STRUCTURE MORE APPROPRIATE FOR THIS IS : TREE STRUCTURE: TAKE_OR_LEAVE_BUT_UNBOUNDED

There are several changes from Coin change. Why ?
They dont an extra row here, I have to revisit this question later 

class Solution:
    def change(self, amount, coins):
        n=len(coins)
        dp = [[0 for _ in range(amount+1)] for _ in range(len(coins)+1)]
        
        ## 0th column fill with 1
        for i in range(len(coins)):
            dp[i][0] = 1
        #print(dp)   
        for i in range(n-1,-1,-1):
            for k in range(1, amount+1):      
                if k-coins[i] >=0 and i+1<=n-1:
                    dp[i][k] = dp[i+1][k] + dp[i][k-coins[i]]
                elif k-coins[i] <0 and i+1<=n-1:
                    dp[i][k] = dp[i+1][k]
                elif i+1>n-1 and k-coins[i]>=0:
                    dp[i][k] = dp[i][k-coins[i]]
                    
        #print(dp)
        return dp[0][amount]






==========================================================================================

909. Snakes and Ladders
### because of the presence of snakes, This question went from tree to graph !!!
### Tree style memo or DP will not work as loops are there. 

## ofcourse greedy wont work because of ladders.

SHORTEST PATH WORD SHOULD IMMEDIATELY TRIGGER BFS IN YOUR MIND!
A lot of work is just converting indices to numbers. This will not just depend on odd or even rows. This was a mistake I made. It also depends on whether m-1 row is odd or even. I had to apply XOR gate. 

Working on numbers is better. Why ? Question doesnt really care about indices. Its about getting to number n**2
not the top left corner. 

class Solution(object):
    logging=False
    def bfs(self,node):
        queue=[node]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()

                ## create neighbors
                neighbors=[]
                for num in range(1,7):
                    if current+num in self.dict1:
                        neighbors.append(self.dict1[current+num])
                    else:
                        neighbors.append(current+num)
                if self.logging: print("current",current)
                if self.logging: print("neighbors",neighbors)

                for neighbor in neighbors:
                    if neighbor<self.n**2 and neighbor not in self.visited and neighbor not in queue:  ##continuation
                        queue.insert(0,neighbor)
                    elif neighbor==self.n**2:       ### ans found condn 
                        return level+1
                
                if self.logging: print("queue",queue)
                if self.logging: print("visited",self.visited)
                self.visited[current]=1
            
            level+=1   
        
        return -1

    def snakesAndLadders(self, board):
        self.visited={}
        self.n=len(board[0])
        self.dict1={}
        
        ## My conversion from index to numbers
        # for i in range(self.n):
        #     for j in range(self.n):
        #         if board[i][j]!=-1:
        #             if (self.n-1)%2==i%2:   ### why this?
        #                 self.dict1[self.n**2-(self.n*i+self.n-j-1)]=board[i][j]
        #                 print("mapped",self.n**2-(self.n*i+self.n-j-1),board[i][j])
        #             else:
        #                 self.dict1[self.n**2-(self.n*i+j)]=board[i][j]      
        
        # BETTER CONVERSION AFTER SOME INSPIRATION
        # SIMPLY LOOPING AROUND AND KEEPING COUNT, REVERTING ALTERNATIVELY
        count=0
        for i in range(self.n-1,-1,-1):
            j_range=range(0,self.n,1) if (self.n-1)%2==i%2 else range(self.n-1,-1,-1) ## J ALTERNATES
            for j in j_range:
                count+=1
                if board[i][j]!=-1:
                    self.dict1[count]=board[i][j]
        return self.bfs(1)                      
==========================================================================================                           
53 Maximum Subarray 

TAG: DYNAMIC Programming, Array
Group : Maximum Product Subarray

(DP question)
No way an easy question although marked so  
TRICKY QUESTION! I forget the logic each time.
So basically max sum can occur at each index in the array and then we have to take the global max. 
and so the dp function f(i) denotes the max subarray sum at index i [for subarrays ending at this point]

Now this maximum sum can only be produced from two cases:
1. by adding current no to the previous maximum sum(Recursive ofcourse) to the left 
    or 
2. just by the nth number 
Realizing this is difficult! Only two cases ( The max subarray grid crosses the boundary of last index or not. If it does then the result is f(n-1)+nth or it doesnÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢t then (nth) )
DP =>        f(n) = Max[ f(n-1)+ n(th) , n(th) ]

f[i] = max Sum that ends at the i(th) element of the array
 f(n) is not the answer. 
The answer  is the global maximum of the dp array.


class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        f=[nums[0]]
        
        for i in range(1,len(nums)):
            f.append(max(f[i-1]+nums[i],nums[i]))
        
        return max(f)

THIS IS A BOTTOM UP DP 
-----------------------------------
This code is from opposite side. 
-----------------------------------

class Solution(object):
    def maxSubArray(self, nums):
        f=[0 for i in range(len(nums))]
        f[-1]=nums[-1]
        
        
        for i in range(len(nums)-2,-1,-1):
            f[i]= max(f[i+1]+nums[i],nums[i])
        
        return max(f)

-----------------------------------
Method2: 0th based index solution like the question subarray sum equals k

   def maxSubArray(self, nums):
        dictMinSum=0
        sum1=0
        maxSum=float("-inf")
        for i in range(len(nums)):
            sum1+=nums[i]
            maxSum=max(maxSum,sum1-dictMinSum)
            dictMinSum=min(dictMinSum,sum1)   ### update AFTER checking not before else it will go wrong
            
        return maxSum
==========================================================================================                           
560. Subarray Sum Equals K
This is a very very tricky question which you have to remember 
1. Idea is to maintain a 0th index based sum
2. A dict maintains the freq of sums 
2. As we iterate we update sum and check for sum-target in the dictionary because if we find it we know that our subarray exists between those two indices
3. We add the current sum to the dict BUTTT after checking not before because the dict maintains a list of things we have seen earlier (if we add current to it then its not right)
3. Sometimes if a given sum appears twice if update frq of that in dict and whenever we are looking for that sum we add the freq because our subarrays can now occur between multiple indices 

class Solution(object):
    def subarraySum(self, nums, k):
        dict1={0:1}
        
        count=0
        sum1=0
        for i in range(len(nums)):
            sum1+=nums[i]
            if sum1-k in dict1:
                count+=dict1[sum1-k]
            if sum1 not in dict1:
                dict1[sum1]=1
            else:
                dict1[sum1]+=1
        return count   
==============================================================================================================
523. Continuous Subarray Sum
The concept is similar to 560. Subarray Sum Equals K
If there is a multiple between two numbers i and j, remainder from sum till i will be same as remainder from sum till j. 
Why ? the perfect multiple will not contribute to remainder. So instead of 0th index sum we keep 0th index sum-remainder. 
Note: 0 is also a multple of k but subarray has to be atleast of length 2 here. 

class Solution(object):
    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
            
            
        dict1={0:-1}   ## this is crucial the sum-remainder 0 is obtained at -1 index
        sum1=0
        for i in range(len(nums)):
            
            sum1+=nums[i]
                    
            if (sum1)%k in dict1 and i-dict1[sum1%k]>=2:   ## second part is important to avoid edge cases 
                return True
            if sum1%k not in dict1:     ### basically we add sum-remainders to dict   
                dict1[sum1%k]=i 
            

==============================================================================================================
152. Maximum Product Subarray
I was able to do this myself. You need to realize that max can be generated by mins too(negatives). So while keeping a track of max at any index we will also keep track of min at any index.
# [-2,3,-2,4]
# f=[-2,3,12,48]
# g=[-2,-6,-6,]
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        f=[None for x in range(len(nums))]
        g=[None for x in range(len(nums))]
        
        f[0]=nums[0]
        g[0]=nums[0]
        for i in range(1,len(nums)):
            f[i]=max(f[i-1]*nums[i],nums[i],g[i-1]*nums[i])
            g[i]=min(f[i-1]*nums[i],nums[i],g[i-1]*nums[i])
        return max(f)
can be optimized because we only need last two 
==============================================================================================================

==============================================================================================================
198. House Robber
# Make a greedy choice for the first index, how does this affect the total. Is greedy the best option? No
# f[0]= max(nums[0]+f[2],f[1])

# f[i] is the total rob sum which you can get from an array starting at i and going till
#         end. 
# f[i]=max(nums[i]+f[i+2],f[i+1])
# base case 
# f[n-1]=nums[n-1]
# return f[0] as answer

class Solution(object):
    def rob(self, nums):
        n=len(nums)
        f=[None for x in range(len(nums))]
        f[n-1]=nums[n-1]
        f[n-2]=max(nums[n-1],nums[n-2])
        
        for i in range(n-3,-1,-1):
            f[i]=max(nums[i]+f[i+2],f[i+1])
            
            
        
        return f[0]
---------------------------------------------------
going upwards 
https://www.youtube.com/watch?v=xlvhyfcoQa4

f(n) = total robbed money till this n 

DP formula -
f(n) = max(n + f(n-2), f(n-1))
possibilities at n, (robbing or not robbing)
Robbing at n -> definitely cannot rob at (n-1) because adjacent , so rob at n & do rob at (n-2) or do not : n+f(n-2)
Not robbing at n -> f(n-1)
==============================================================================================================
213. House Robber II
basically If i rob at 0, cant rob at n-1 for sure.    ### we need a dp which hasnt robbed at n-1
But if I dont rob at 0, i may rob at n-1 or may not.  ### regular house robber dp

So I separate with two cases/dps, I do not rob at n-1           : g ## this is the special dp i keep extra
                              I may rob may not rob at n-1      : f ## regular dp
Now, calculate till 2 
at 1, its either rob1+didnt rob at n-1 so nums[1]+g[2] or didnt rob 1 so f[1]  

class Solution(object):
    def rob(self, nums):
        n=len(nums)
        if n==1:
            return nums[0]
        if n==2:
            return max(nums[0],nums[1])
        
        f=[0 for i in range(n)]
        g=[0 for i in range(n)]
        f[n-1]=nums[n-1]
        g[n-1]=0
        f[n-2]=max(nums[n-2],nums[n-1])
        g[n-2]=nums[n-2]
        
        for i in range(len(nums)-3,0,-1):
            f[i]=max(nums[i]+f[i+2],f[i+1])
            g[i]=max(nums[i]+g[i+2],g[i+1])
        #print(f)
        #print(g)
        return max(nums[0]+g[2],f[1])
---------------------------------------------------
Optimzing in space using variables
class Solution(object):
    def rob(self, nums):
        n=len(nums)
        f2=nums[n-1]
        g2=0
        f1=max(nums[n-2],nums[n-1])
        g1=nums[n-2]
        
        for i in range(len(nums)-3,0,-1):
            f0=max(nums[i]+f2,f1)
            g0=max(nums[i]+g2,g1)
            f1,f2=f0,f1
            g1,g2=g0,g1

        return max(nums[0]+g2,f1)
==============================================================================================================
337. House Robber III
This is basically a head recursion
MAKE THE BLOODY CHOICE, DO YOU WANT TO INCLUDE THE root OR NOT INCLUDE THE root
I missed a case in without, otherwise solved it on my own, if left and right return the sums with and without root.
The question can be solved at the root.

class Solution(object):
    def robR(self, root):
        if not root:
            return 0,0
        if not root.left and not root.right:
            return root.val,0
        
        withLeftRoot,withoutLeftRoot=self.robR(root.left)
        withRightRoot,withoutRightRoot=self.robR(root.right)
        
        withoutRoot=max(withLeftRoot+withRightRoot,
                        withLeftRoot+withoutRightRoot,
                        withoutLeftRoot+withRightRoot,withoutLeftRoot+withoutRightRoot) #withoutLeftRoot+withoutRightRoot missed this case
        
        
        withRoot=withoutLeftRoot+withoutRightRoot+root.val
        return withRoot,withoutRoot
    
    
    def rob(self, root):
        #print(self.robR(root))
        return max(self.robR(root))



==============================================================================================================
256. Paint House
Tags: DP 
Tricky question to think about :
I thought like this 
## I can see the tree of choices
# # Brute force is simply doing all the calculations. Does anything repeat? Yes subtrees have same elements. Answer will be same if you go for color x and house i because subtrees after that will look similar(all possibilities) and produce single minimum. 
# so 
### WE CAN DO BRUTE FORCE DFS

### HOW ABOUT WE BREAK IT DOWN IN BottomUpTabulation
# totalMin=min(cost(1,1)+cost(1,2)+cost(1,3)) i is house and j is color
# cost(1,1)=min(cost(2,2),cost(2,3))

# cost(i,1)=min(cost(i+1,2),cost(i+1,3))
# Base cases
# cost(n,1) ## known
# cost(n,2)
# cost(n,3)
---------------------------------------------------------------------
## general way to think, make a choice and then see if you still calculate the totalgoal after that choice
---------------------------------------------------------------------

f[i] is the minimum cost to paint all houses starting from index i and going till end
ans=min(cost[0][1]+min(f[1][2],f[1][3]),cost[0][2]+min(f[1][1],f[1][2]),cost[0][3]+min(f[1][2],f[1][1]))

f[i][j] is the minimum cost to paint remaining houses if we use j color for house i


## Bottom up with tabulation 

class Solution(object):
    def minCost(self, costs):
        n=len(costs)
        f=[[float("inf") for x in range(3)] for i in range(n)]
        
        f[n-1][0]=costs[n-1][0]
        f[n-1][1]=costs[n-1][1]
        f[n-1][2]=costs[n-1][2]
        
        
        for i in range(n-2,-1,-1):
            f[i][0]=min(f[i+1][1],f[i+1][2])+costs[i][0]
            f[i][1]=min(f[i+1][0],f[i+1][2])+costs[i][1]
            f[i][2]=min(f[i+1][0],f[i+1][1])+costs[i][2]
        
        return min(f[0]) 
## note tabulation can be improved if we simply overwrite costs, its not needed once its used.
---------------------------------------------------------------------
isnt that simple
Think about minimum cost of painting a house with a given color.
It will be the cost + previous house cannot be painted with the given color(take min for the othr two colors).If you dont see the tree of all possibilities this is difficult to think of.
Boom Recursion. Bottom up 
---------------------------------------------------------------------

class Solution(object):
    def minCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        ##logic
        ##syntax ok
        ##edge 
        #empty ok 
        if len(costs)==0: 
            return 0
        
        for i in range(1,len(costs),1):
            costs[i][0]=costs[i][0]+min(costs[i-1][1],costs[i-1][2])
            costs[i][1]=costs[i][1]+min(costs[i-1][0],costs[i-1][2])
            costs[i][2]=costs[i][2]+min(costs[i-1][0],costs[i-1][1])
    
        return min(costs[-1])
======================================================================
118. Pascal's Triangle
Simple for loop solution, 
They also calling it DP because nth row uses previous row to generate. Ok!
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        ans=[[1],[1,1]]
        
        if numRows==1:
            return [ans[0]]
        if numRows==2:
            return ans   
        for i in range(3,numRows+1,1):
            nR=[]
            nR.append(1)
            for j in range(1,len(ans[-1]),1):
                nR.append(ans[-1][j]+ans[-1][j-1])
            nR.append(1) 
            ans.append(nR)
        return ans 
======================================================================
 Pascal's Triangle II
class Solution(object):
    def getRow(self, rowIndex):
        prev=[1,1]
        if rowIndex==0:
            return [1]
        if rowIndex==1:
            return prev
        
        for j in range(1,rowIndex,1): ##runs rowIndex-1 times
            nR=[1]
            for k in range(1,len(prev),1):
                nR.append(prev[k]+prev[k-1])
            nR.append(1)
            prev=nR
        return prev
======================================================================
55. Jump Game
## greedy wont give us the answer
## what does the possibility tree look like?
## made this, bfs looks like a good approach
## visited optimization will make sense as if you cant reach from one index once
## no need to try again

## easy and intuitive BFS algorithm, Note : no need of levels here, DFS will also give answer.
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        n=len(nums)
        queue=[0]
        level=0
        visited={}
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                if current==n-1:
                    return True  
                neighbors=[current+jump_score for jump_score in range(1,nums[current]+1,1)]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        if neighbor==n-1:
                            return True  
                        queue.insert(0,neighbor)
                        visited[neighbor]=1
                
                
            level+=1
        return False 
---------------------------------------------------------------------
BOTTOM UP TABULATION APPROACH - TLE
## general way to think, make a choice and then see if you still calculate the totalgoal after that choice
# ### bottom up DP
# f[i] indicates true or false for array starting at i and till end 
# our answer : f[0]
# f[0]= True if WITHIN REACH any f[i] is True     
# base case 
# i=n-1 true
# i=n-2     
    
class Solution(object):
    def canJump(self, nums):
        n=len(nums)
        f=[None for x in range(n)]
        f[n-1]=True
        for i in range(n-2,-1,-1):
            for j in range(1,nums[i]+1):
                if  f[i+j]==True:
                    f[i]=True
                    break
            else:
                f[i]=False
                
        return f[0]
Time complexity : O(n^2). 
For every element in the array, say i, we are looking at the next nums[i] elements to its right aiming to find a True index. nums[i] can be at most n, where n is the length of array nums.
Space complexity : O(n)
---------------------------------------------------------------------
Further optimization -- In the previous solution we see that while going back we just need to if check if from ith 
any of the True can be reached and we break on first reachable True. so in fact we just care about first reachable True (leftmost True).
we can keep track of that separately and leftmostindex to i if it is reachable from i 

This approach is called GREEDY. Why we reduced a TREE STRUCTURE and going in all directions to just check along a branch GREEDILY. WE ARE LUCKY THAT THIS SOLUTION EXISTED BUT THERE IS NO GAURANTEE THAT IT WILL ALWAYS EXIST. GREEDY HAPPENED IN BottomUPTabulation here. Is it also possible in DFS/BFS tree traversal.


class Solution(object):
    def canJump(self, nums):
        n=len(nums)
        leftmostTrueIndex=n-1
        for i in range(n-2,-1,-1):
            if i+nums[i]>=leftmostTrueIndex :
                leftmostTrueIndex=i
        return leftmostTrueIndex==0
Time complexity : O(n) ## single check instead of iteration
---------------------------------------------------------------------
Alternate iterative approach 
So for each index we have to check whether we can reach here by comparing i with maxindex if not return False and we keep a track of max index we can jump to at each index if this exceeds last index then return True

class Solution:
    def canJump(self, nums):
        maxIndex=0
        for i in range(len(nums)):
            if i>maxIndex:          ##can we reach here ? at any point this happens means we cant reach here
                return False
            if maxIndex>=len(nums)-1:  ###checking maxindex at every iteration it it has exceeded
                return True
            maxIndex=max(maxIndex,i+nums[i])
        return True  # for case of empty array
======================================================================
45. Jump Game II
Now instead of just saying if we can reach or not. we need to return the min steps to reach.
Ofcourse BFS should give the answer. What is the time complexity? Exponential?

class Solution(object):
    def jump(self, nums):
        n=len(nums)
        queue=[0]
        level=0
        visited={}
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                if current==n-1:
                    return 0  
                #print(current,nums[current])
                neighbors=[current+jump_score for jump_score in range(1,nums[current]+1,1)]
                #print(neighbors)
                for neighbor in neighbors:
                    if neighbor==n-1:
                        return level+1      
                    queue.insert(0,neighbor)
                    visited[neighbor]=1
            level+=1
        return False 
---------------------------------------------------------------------
Do we have another GREEDY SOLUTION FOR THIS ?
yes? while going backwards in bottomUpTabulation we can simply greedily choose the min cost. Now isnt the question is similar to "min cost Climbing stairs"

class Solution(object):
    def jump(self, nums):
        n=len(nums)
        f=[float("inf") for x in range(n)]
        f[n-1]=0
        for i in range(n-2,-1,-1):
            if nums[i]!=0:                      ### min arg doesnt work with empty lists to corrected this edge case
                f[i]=min(f[i+1:i+nums[i]+1])+1  ## otherwise its a simple minimum of the NEXT JUMP ARRAY
        return f[0]



======================================================================
62. Unique Paths --> similar to Climbing Stairs
## bottom up tabulation
# f[i][j]=f[i-1][j]+f[i][j-1]
# Base cases
# f[0][0]=1
# f[0][1]=1
# f[1][0]=1
class Solution(object):
    def uniquePaths(self, m, n):
        f=[[None for x in range(n)] for x in range(m)]
        for i in range(m):
            for j in range(n):
                if i==0: 
                    f[0][j]=1
                elif j==0:
                    f[i][0]=1
                else:
                    f[i][j]=f[i-1][j]+f[i][j-1]
        return f[m-1][n-1]
======================================================================
64. Minimum Path Sum ### similar to min cost climbing stairs in 2D
# f[i][j]=min(f[i][j-1],f[i-1][j])+grid[i][j]
# return f[m-1][n-1]
class Solution(object):
    def minPathSum(self, grid):
        m=len(grid)
        n=len(grid[0])
        for i in range(m):
            for j in range(n):
                if i==0:
                    if j==0:
                        pass
                    else:
                        grid[i][j]=grid[i][j-1]+grid[i][j]
                elif j==0:
                    if i==0:
                        pass
                    else:
                        grid[i][j]=grid[i-1][j]+grid[i][j]
                else:
                    grid[i][j]=min(grid[i][j-1],grid[i-1][j])+grid[i][j]
        return grid[m-1][n-1]
=======================================================================================================
120. Triangle
I realize by using a simple example how at each level i can calculate min bu using previous level
Now its just a matter a getting loop right 
# 2
# 5,6
# 11,10,11
# 14,11,18,14
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        m=len(triangle)
        cR=triangle[0]
        for i in range(m-1):                      #loop runs m-1 times
            nR=[None for x in range(len(cR)+1)]   ##new Row is one more than previous row
            for j in range(len(nR)):              ## now fill it up 
                if j==0 :
                    nR[0]=cR[0]+triangle[i+1][0]
                elif j==len(nR)-1:
                    nR[len(nR)-1]=cR[-1]+triangle[i+1][-1]
                else:
                    nR[j]=min(cR[j],cR[j-1])+triangle[i+1][j]
            cR=nR                               ##dont forget this
        
        return min(cR)
---------------------------------------------------------------------
Optimize by rewriting triangle
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        m=len(triangle)
        for i in range(1,m,1):   ### dont start at 0 otherwise you will use the last row
            for j in range(i+1):
                if j==0:
                    triangle[i][j]+=triangle[i-1][j]
                elif j==i:
                    triangle[i][j]+=triangle[i-1][-1]
                else:
                    triangle[i][j]+=min(triangle[i-1][j],triangle[i-1][j-1])
        return min(triangle[-1])
======================================================================
91. Decode Ways
Is single breakdown valid? add neighbor
is double breakdown valid? add neighbor

There are no edge cases in this question contrary to popular belief. LOL. 
Just need to decide whats the base case properly. 
I got confused in this. Is single digit & double digit base case? No because we still need to check validity. 
Also I need to breakdown double digit. So that cant be base case. Empty string is base case. 

Now I made the mistake of returning 0 when string empty. Why did i think this. I thought what is the number of ways to create decode when string is empty ->0. 
But the thought process is if we have succeeded in creating empty string, the tree path we were following was SUCCESSFUL. 
and we are counting number of successful TREE PATHS.
TYPE: REACH THE END AND TELL ME IF THIS TREE PATH WAS VALID. 
Do we have other questions like this?

## simple recursion without memo will be TLE

### TOP DOWN MEMOIZED RECURSION 
# 1. Brute Force
# #         "120"
# # A"2"            C

# 226

# f(0) = I(1)*f(1)+I(2)*f(2)
# f(i) = I(1)*f(i+1)+I(2)*f(i+2)
# f("12")=f("2")+f("12")
# = f("")+ 

# What is the base case?
# i=n empty string f(i)=0   WRONG f(n)=1

# 2. Optimized
# 3. Edge cases 
# 4. Syntax
class Solution(object):
    logging=False
    def numDecodingsR(self, root):
        ## base case
        if root=="":            ## base case mistakes made here 
            return 1
        if root in self.memo:
            return self.memo[root]
        
        # neighbors 
        neighbors=[]
        if root[0] in "123456789":          ### validity of single 
            neighbors.append(root[1:])
        if int(root[:2])>=10 and int(root[:2])<=26: ### validity of double
            neighbors.append(root[2:])
        
        root_list=[]
        for neighbor in neighbors:
            neighbor_ans=self.numDecodingsR(neighbor)
            root_list.append(neighbor_ans)
        
        if self.logging: print(root)
        if self.logging: print(neighbors)
        
        root_ans=sum(root_list)
        self.memo[root]=root_ans
        
        return self.memo[root]

    def numDecodings(self, s):
        self.memo={}
        return self.numDecodingsR(s)
---------------------------------------------------------------------
Instead of carrying the string I can also just carry the index
class Solution(object):
    logging=False
    def numDecodingsR(self,root):
        ## base case
        if root==len(self.s):
            return 1
        if root in self.memo:
            return self.memo[root]
        
        # neighbors 
        neighbors=[]
        if self.s[root] in "123456789":
            neighbors.append(root+1)
        if int(self.s[root:root+2])>=10 and int(self.s[root:root+2])<=26:
            neighbors.append(root+2)
        
        root_list=[]
        for neighbor in neighbors:
            neighbor_ans=self.numDecodingsR(neighbor)
            root_list.append(neighbor_ans)
        
        if self.logging: print(root)
        if self.logging: print(neighbors)
        
        root_ans=sum(root_list)
        self.memo[root]=root_ans
        
        return self.memo[root]
            
    def numDecodings(self, s):
        self.memo={}
        self.s=s
        return self.numDecodingsR(0)
---------------------------------------------------------------------
Bottom up DP tabulation 
# f[i] is the number of ways to decode a string starting at index i and going till end
# f[i]=I(s[i])*f[i+1]+I(s[i:i+2])*f[i+2]
s[i:i+2] should be valid two digits
s[i] should be valid one digit

Defining base case is crucial. i=n and DP[n]=1
CRUCIAL AND STANDARD IN DP PROBLEMS, otherwise it becomes very messy.


class Solution(object):
    logging=False
    def numDecodings(self, s):
        n = len(s)
        dp = [None for i in range(n+1)]     ### KEEPING EXTRA IS CRUCIAL AND STANDARD IN DP PROBLEMS
        dp[n] = 1                            
        for i in range(n-1,-1,-1):
            indicator1 = 1 if s[i] in "123456789" else 0
            indicator2 = 1 if int(s[i:i+2])>=10 and int(s[i:i+2])<=26 else 0
            
            if i+2<=n:                      ### BOUNDARY CHECK IS BASIC AND STANDARD FROM THE BOTTOM UP METHOD
                dp[i] = indicator1*dp[i+1]+indicator2*dp[i+2]
            else:
                dp[i] = indicator1*dp[i+1]
            
        print(dp)
        return dp[0]
        
======================================================================
300. Longest Increasing Subsequence
Compare this to "Longest Consecutive Sequence" - This is a sequence where neighbors are clearly defined and checked 
                                               - 2nd indexing doesnt matter here which matters in this problem
## first you need to realize to use DP.
## Why DP? "SUBSEQUENCE" not substring or subarray
## SUBSEQUENCES mean you can choose or not choose, which makes it exponential 
## now when we have exponential we will try to make it linear or quadratic
## How ? bottom up DP with tabulation
## break it baby
## how to break it?
## make a choice and then evaluate the original answer with respect to the choice
## Lets say we start the subsequence with index 0
## f[i] indicates subsequence starting at i
## f[i]=max(1+f(i+1)*I(i+1),f(i+2)*I(i+2),f(n-1)*I(n-1) ) ## only to forward indices
## we can only go to those indices which nums is larger so indictor is deciding that
## now whats the base case? base case is last element, f[n-1]=1

class Solution(object):
    def lengthOfLIS(self, nums):
        f=[float("-inf") for i in range(len(nums))]
        f[len(nums)-1]=1
        for i in range(len(nums)-2,-1,-1):
            for j in range(i+1,len(nums),1):
                indicator=1 if nums[j]>nums[i] else 0
                f[i]=max(f[i],1+f[j]*indicator)

        return max(f)       ## dont return f[0] lol!
Time complexity : O(N2) ## for every index we iterate to the right and check
Space: O(N) for size of f

Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?
Some stupid greedy solution is available. lol
https://www.youtube.com/watch?v=22s1xxRvy28 ##Patience sort explained
https://leetcode.com/problems/longest-increasing-subsequence/discuss/1326308/C%2B%2BPython-DP-Binary-Search-BIT-Solutions-Picture-explain-O(NlogN)
======================================================================
221. Maximal Square
Here I how i got the solution on my own. :)
1. What uniquely identifies a sqaure ? top left cordinate and len 
2. So we are simply asked out of all top left cordinates which returns the maximum 
3. Now brute force is obvious of course that you go to each top left and start expanding 
4. Can we optimize? what if i start from the bottom , for size to go to 2 all neighbors should have maximal 1!!
5. Thats it , replace the array with maximal square areas. Take the global max
6. Going back wards will be a pain what if we go forward. , take botttom right corners
7. whats the function to determine maximal square value ?? We take the min of neighbors and add it to i,j value 
BUT only if i,j is 1. So i cleverly used value of i,j as indicator function. TOOK LOT OF EXAMPLES TO SEE THIS!
8. we only do this operation where its possible by cordinates, other wise we dont need to do anything , pass
9. But we need to do global max check at all cordinates even where this operation isnt happening.

class Solution(object):
    def maximalSquare(self, matrix):
        globalmax=0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i-1>=0 and j-1>=0:
                    matrix[i][j]=(min(int(matrix[i-1][j]),int(matrix[i][j-1]),int(matrix[i-1][j-1]))+int(matrix[i][j]))*int(matrix[i][j])
                globalmax=max(globalmax,int(matrix[i][j]))       
        return globalmax**2


======================================================================
1143. Longest Common Subsequence
## Brute Forcing - simple pointers, dictionary
## 1. For evry index in text2, check if it exists in text1. if no move forward
## if yes start the window there and move forward while checking for next
## No even this doesnt work because the other one also subsequence
## How many "Subsequence" can be created from a string of length n ? list of 2^n 
## because at every element we have a choice to leave or take
## from the other one list of 2^n1
## compare these two lists (2^n)*(2^n1): Time complexity

## 2. Sliding window can have issues--
### There are two strings -- SLW usually has one string
##  now once i see exponential time in brute force, i dont want to think about SLW
THIS HAS THE WORD "SUBSEQUENCE"-- FORGET SLIDING WINDOW!!

## 3. DP 
To convert exponential to non-exponential we have - greedy or DP.
However, if a greedy algorithm exists, then it will almost always be better than a dynamic programming one. 
What is greedy? It is simply taking the best step at each time while descending the tree structure.

While it's very difficult to be certain that there is no greedy algorithm for your interview problem, over time you'll build up an intuition about when to give up. You also don't want to risk spending so long trying to find a greedy algorithm that you run out of time to write a dynamic programming one (and it's also best to make sure you write a working solution!).
We do not know greedy so let me just make up the tree. 

How does the tree look like ? Made the tree
Got the recurrence:
f(i,j)=f(i+1,j+1)+1 if characters match, this is not a decision. This is necessary, dont forget adding +1
f(i,j)=max(f(i,j+1), f(i+1,j)) whenever chars dont match we increment both sides and check max in both
made the Tree and verified. 
What is the base case, base case is whenever either i or j is n1 or n2. 

ALGO_NAME:BOTTOM_UP_TABULATION_2D_BACKWARDS 
LOGIC POINTS
1. Decide you will use range or indexes to iterate on rows and columns based on recurrence reln.
   Here:  
   Range for rows and columns is index so both indexes. 
2. Fill the base cases first.
    Here the nth row can be filled and the nth column can be filled. 
3. Decide what is the sequence of calculation.
    Of course we need to go backward in rows.
    We also need to go  backward in columns. Because we use right side elements to fill left side.
4. Out of bounds check: Can we go out of bounds? Cant go out of bounds here
SYNTAX POINTS
5. Conversion needed? When we are in dp[i][k]. No conversion is needed here because we deal with indexes directly.


class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        n1=len(text1)
        n2=len(text2)
        dp=[[None for j in range(n2+1)] for i in range(n1+1)]       ### took extra as always 
        
        ## fill n2th column   ## base cases are easy to think about here
        for i in range(n1+1):
            dp[i][n2]=0
        ## fill n1th row 
        for j in range(n2+1):
            dp[n1][j]=0
            
        #print(dp)
    
        for i in range(n1-1,-1,-1):
            for j in range(n2-1,-1,-1):
                ## use recurrence relationship
                if text1[i]==text2[j]:
                    dp[i][j]=dp[i+1][j+1]+1
                else:
                    dp[i][j]=max(dp[i+1][j],dp[i][j+1])
        #print(dp)
        
        return dp[0][0]
I was so scared of this question and then i did it myself. 
---------------------------------------------------------------------
We should be able to do it with TREE_STYLE_DFS_HEAD_RECURSIVE
======================================================================
97. Interleaving String
I did this on my own after struggling for some time with base cases.
aabcc, dbbca
aadbbcbcac

How is this question different from longest common subsequence
Its kind of the opposite 
 d b b c a
a
a
b
c
c
Recurrence relationship was perfectly captured
f(i,j,k)=either f(i+1,j,k+1) if i and k match   ### single child 
            //  f(i,j+1,k+1) if j and k match
else:
    f(i,j,k) = f(i+1,j,k+1) OR f(i,j+1,k+1) if both match   ## binary child 


After this i realized k is simply i+j

if no match return False
    
Base Case:
i=n1 and j=n2 True, ALL OTHERS WE NEED TO FILL !!!!

Started getting confused in how to deal with base cases. 
THERE IS ONLY ONE BASE CASE HERE? WHY? i==n1 and j==n2 and dp value True.
Rest lets say s1="" and s2="abc" s3="abc"
Answer should be True
Now when you go to calculate, dp[0][0] i cant match but j matches i+j so we are supposed to move to the right.
Here if mismatch happens, you dont have to "return" FALSE. JUST ASSIGN DP GRID AS FALSE
dp[0][0]=dp[0][1]=dp[0][2]=dp[0][3]=dp[0][4] which is True. if it mismatches at any point in the middle we just put it 
as False


1. i==n1 and j==n2 needed to be handled separately due to index errors


class Solution(object):
    logging=False
    def isInterleave(self, s1, s2, s3):
        n1=len(s1)
        n2=len(s2)
        if len(s3)!=n1+n2:
            return False
        
        dp=[[None for j in range(n2+1)] for i in range(n1+1)]
    
        for i in range(n1,-1,-1):
            for j in range(n2,-1,-1):
                if self.logging: print("i,j",i,j)
                if i==n1 and j==n2:
                    if self.logging: print("n1n2True")
                    dp[i][j]=True
                elif i==n1:
                    if self.logging: print("comparing",s2[j:],s3[i+j:])
                    if s2[j]==s3[i+j]:
                        dp[i][j]=dp[i][j+1]
                    else:
                        if self.logging: print("n1False")
                        dp[i][j]=False
                elif j==n2:
                    if self.logging: print("comparing",s1[i:],s3[i+j:])
                    if s1[i]==s3[i+j]:
                        dp[i][j]=dp[i+1][j]
                    else:
                        if self.logging: print("n2False")
                        dp[i][j]=False
                
    
                elif s1[i]==s3[i+j] and s2[j]==s3[i+j]:
                    if self.logging: print("both equal",s1[i:],s2[j:],s3[i+j:])
                    dp[i][j]=dp[i+1][j] or dp[i][j+1]
                elif s2[j]==s3[i+j]:
                    if self.logging: print("j equal",s1[i:],s2[j:],s3[i+j:])
                    dp[i][j]=dp[i][j+1]
                elif s1[i]==s3[i+j]:
                    if self.logging: print("i equal",s1[i:],s2[j:],s3[i+j:])
                    dp[i][j]=dp[i+1][j]
                else:
                    if self.logging: print("mismatch")
                    dp[i][j]=False      ## DONT MAKE THE STUPID MISTAKE OF RETURNING FALSE
                    ### WE CALCULATE ALL POSSIBLE COMBINATION SO SEVERAL WILL BE FALSE
                    ### WE ONLY CARE ABOUT DP 0 0 
        
        #print(dp)
        
        return dp[0][0]   
======================================================================
115. Distinct Subsequences
This question is categorised as HARD, but it came so easily to me, solved in 15 mins

subsequences shout DP 
looks like 0/1 knapsack 
I see a tree with two brances at every index. 
# but can we take the letter always ? no
leaves tell me what i formed. I compare leaf with t thats it. 
ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION 
TREE STRUCTURE: TAKE_OR_LEAVE_01KNAPSACK

ALGO_NAME: bottom_up_tabulation_

Recurrence reln 
f(0,0)=matches then choose to take or not take f(1,1)+f(1,0)
        no_match f(1,0)
f(i,j)--matches f(i+1,j+1)+f(i+1,j)
        no match f(i+1,j)
Base cases:
i=n1 s1="" and t="abc"  0 but if t="" then 1
So entire row of i=n1 is 0 in DP,but corner= 1 
so i=n1 and j=n2  -- 1 
i=any j=n2  s1="abc" and t="" 1
entire column=1 for j=n2

class Solution(object):
    def numDistinct(self, s, t):
        n1=len(s)
        n2=len(t)
        dp = [[None for j in range(n2+1)] for i in range(n1+1)]
        ## row n1 fill
        for j in range(n2+1):
            dp[n1][j]=0
        ## column n2 fill
        for i in range(n1+1):
            dp[i][n2]=1
        ## corner became 1 automatically after this 
        
        for i in range(n1-1,-1,-1):
            for j in range(n2-1,-1,-1): ## i went backwards because we need right members
                if s[i]==t[j]:
                    dp[i][j]=dp[i+1][j+1]+dp[i+1][j]
                else:
                    dp[i][j]=dp[i+1][j]
        print(dp)
        return dp[0][0]
======================================================================
72. Edit Distance

DID IT OWN MY OWN!!!

Went down the wrong alley
upon reading, i can immediately tell this is a graph problem 
Why ? Is this not like word ladder? here more ops are allowed. 
why a graph and not a tree? because it can loop back 
so i definitely need a visited thing to avoid loops
now BFS on the graph will give me answer
defining neighbors will be a pain 

"abc" -> "abb"->"abd"->"abc"
--------------------------------------
# Apparently a much simpler approach is possible using ALGO_NAME: BOTTOM_UP_TABULATION_2D
WHY WORD LADDER CANT BE SOLVED LIKE THIS?? WORD LADDER IS A GRAPH
WHILE THIS IS A TREE


 h o s 
h
o
r      1
     1 0

Base cases: (simply think about what happens in your rightmost column and bottommost row)
D between "" and "a" is 1 so i==n1 and j==n2-1 is 1, single cell
D between "a" and "" is 1 so i==n1-1 and j=n2 is 1, single cell
D between "" and "abc" is 3 so if i==n1, D is simply len string-- side row and side column done 
D between "" and "" is 0 so i==n1 and j==n2 is 0  corner done
D between same chars is 0 

See the tree at each index, i take a decision to leave/take as is/take but change/
what about inserts?
lets no consider different lengths

match : f(0,0)= f(1,1) -> f(i,j)=min(f(i+1,j+1),1+f(i,j+1),1+f(i+1,j))
no match : f(0,0)=min(1+f(1,1),1+f(0,1),1+f(1,0))

"horse"
"ros"
hor
mhor
1=min(f(or,hor),1+f(hor,hor) ,1+f(or,hhor)) --"hor"&"hhor"
1=min(1+f(or,hor),1+f(hor,hor) ,1+f(or,hhor))

STRING DP PROBLEMS HAVE A CENTRAL THEME: MATCH? NOMATCH?
WHAT ARE THE CHILDREN IN EACH CASE. The only couple of things you can do its increase either or both
TAKE SOME EXAMPLE TO FIGURE IT OUT. 

class Solution(object):
    def minDistance(self, word1, word2):
        n1=len(word1)
        n2=len(word2)
        
        dp=[[None for j in range(n2+1)] for i in range(n1+1)]
        
        for i in range(n1,-1,-1):
            for j in range(n2,-1,-1):
                if i==n1 and j==n2:
                    dp[i][j]=0
                elif i==n1:
                    dp[i][j]=len(word2[j:])
                elif j==n2:
                    dp[i][j]=len(word1[i:])
                else:
                    if word1[i]==word2[j]:  ##match case 
                        dp[i][j]= min(dp[i+1][j+1],1+dp[i][j+1],1+dp[i+1][j])
                    else:                   ## mismatch
                        dp[i][j]= min(1+dp[i+1][j+1],1+dp[i][j+1],1+dp[i+1][j])
                        
        #print(dp)             
        return dp[0][0]
    
=====================================================================
10. Regular Expression Matching
VERY VERY PAINFUL QUESTION. TOOK HOURS AFTER HELP 
1. FOLLOWING SALIENT POINTS AFTER LEARNING
2. Last edge row has to be dealt with separately. 
3. Mistake is to wait till j reaches *, have to deal with by checking j+1


# Examples
# "abc" "ab."  -- True
# "abc" "ab*"  -- False

# Base cases"
# "" "" -> True       Corner done

# "" "abc" -> False   n1 row 
# "" "."   -> False   n1 row  
# "" ".bc"   -> False   n1 row  
# "" "b*"   -> TRUE
# "" "*"   -> cant appear as given by question but we still need it !

## n2 column "abc" ""  False Corner column done

# Recurrence reln: 
#     match:
#         s[i]==p[j] or (i<=n1-1 and p[j]==".")     CORRECT
#         f(i,j)=f(i+1,j+1)
        
#         i<=n1 and p[j]=="*"                       WRONG 
#             if s[i]==p[j-1]:
#                 f(i,j)= f(i+1,j)
#             else:
#                 f(i,j)= False
            
#     no match
#         f(i,j)=False


class Solution(object):
    logging=True
    def isMatch(self, s, p):
        n1=len(s)
        n2=len(p)
        dp=[[None for j in range(n2+1)] for i in range(n1+1)]
        print(dp)
        
        ### n2 column always False
        if j==n2:
            dp[i][n2]=False
        
        dp[n1][n2]=True
        
        for i in range(n1,-1,-1):
            for j in range(n2-1,-1,-1):
                #if self.logging: print("comparing",s[i:],"&",p[j:])
                
                match = i <= n1-1 and (s[i] == p[j] or p[j] == ".")   
                
                if j+1>len(p)-1 or p[j+1]!="*":
                    if match:
                        if self.logging: print("matched,now comp",s[i+1:],"&",p[j+1:])
                        dp[i][j]=dp[i+1][j+1]
                    else:
                        dp[i][j]= False 

                else:   
                    ##p[j+1] is "*"
                    #print("here")
                    ## https://www.youtube.com/watch?v=HAA8mgxlov8&t=306s logic for below recurrence is here 
                    ## difficult to understand
                    if match:
                        dp[i][j]=dp[i][j+2] or dp[i+1][j]
                    else:
                        dp[i][j]=dp[i][j+2]
                
        #print(dp)
        return dp[0][0]
                    
# class Solution:
#     def isMatch(self, s: str, p: str) -> bool:
#         cache = [[False] * (len(p) + 1) for i in range(len(s) + 1)]
#         cache[len(s)][len(p)] = True
        
#         for i in range(len(s), -1, -1):
#             for j in range(len(p) - 1, -1 ,-1):
#                 match = i < len(s) and (s[i] == p[j] or p[j] == ".")
                
#                 if (j + 1) < len(p) and p[j + 1] == "*":
#                     cache[i][j] = cache[i][j + 2]
#                     if match:
#                         cache[i][j] = cache[i + 1][j] or cache[i][j]
#                 elif match:
#                     cache[i][j] = cache[i+1][j+1]
                    
#         return cache[0][0]               
======================================================================
123. Best Time to Buy and Sell Stock with Cooldown
I am able to make the tree structure on my own.

ALGO : TREE_STYLE_DFS_ITERATIVE_BACKTRACK -- TLE 
Its obvious we can do this. 
### i think 3 variables in state space will be enough 
### I had to add another variable in the state space for cooldown cases. 
The problem is deciding variables in the state space. 
Obviously index is going to be one. Now what are the decisions we have to make after we make the first choice. 
1. We can buy 2. We can sell 3. We can do nothing
Also this is a BINARY TREE as we cant buy and sell at the same time. 
Looks like three options but we are fluctuate between two options, can_buy: boolean. I was able to figure this out on my own. 
We also need to carry a current balance in state space. 
When we go from can_buy : True to False. We just bought. We need to subtract the current number 
When we go from can_buy False to True. We just sold and We need to add the current number. So I on the sign of cool_buy i create an indicator of -1 and 1.
Now cooldown needed to be added: if prev_can_buy was False and can_buy is True. We just sold.
and in this case only one child is available. 

ALGO : TREE_STYLE_DFS_ITERATIVE_BACKTRACK -- TLE 

class Solution(object):
    logging=False
    def dfs(self,prices):
        stack=[(-1,True,0,True)]    ## I start with previous Can_buy TRUE so as to not introduce cool down 
        while stack:
            index,can_buy,profit,prev_can_buy=stack.pop()
            indicatorF=-1 if can_buy==True else 1
            nextNeed_cooldown=True if can_buy==False else False
            if can_buy and not prev_can_buy: ## we just sold if prevCB is False and current is True
                neighbors=[(index+1,can_buy,profit,can_buy)]  ### dont miss case of cooldown
            else:
                neighbors=[(index+1,can_buy,profit,can_buy),(index+1,not(can_buy),profit+indicatorF*prices[index+1],can_buy)] 
            
            if self.logging: print("current", index,can_buy,profit,need_cooldown)
            if self.logging: print("neighbors", neighbors)
            for neighbor in neighbors:
                nextIndex,nextCan_buy,nextProfit,need_cooldown=neighbor
                if nextIndex<len(prices)-1:                ## continuation condn on next
                    stack.append(neighbor)
                else:
                    if self.logging: print("leaf node", neighbor)
                    self.maxProfit=max(self.maxProfit,nextProfit)
    
    def maxProfit(self, prices):
        self.maxProfit=float("-inf")
        self.dfs(prices)
        return self.maxProfit
---------------------------------------------------------------------
ALGO: TREE_STYLE_DFS_TAIL_RECURSIVE --- THIS IS THE SAME AS TREE_STYLE_DFS_ITERATIVE_BACKTRACK, so there isnt a point of learning this

class Solution(object):
    logging=False
    def dfsR(self,root,prices):
        index,can_buy,profit,prev_can_buy=root
        
        ##find neighbors
        indicatorF=-1 if can_buy==True else 1
        nextNeed_cooldown=True if can_buy==False else False
        if can_buy and not prev_can_buy: ## we just sold if prevCB is False and ours is True
            neighbors=[(index+1,can_buy,profit,can_buy)]  ### dont miss case of cooldown
        else:
            neighbors=[(index+1,can_buy,profit,can_buy),(index+1,not(can_buy),profit+indicatorF*prices[index+1],can_buy)] 
        ##find neighbors done
        for neighbor in neighbors:  ### FIFO 
            nextIndex,nextCan_buy,nextProfit,need_cooldown=neighbor
            if nextIndex<len(prices)-1:                ## continuation condn on next
                self.dfsR(neighbor,prices)
            else:
                if self.logging: print("leaf node", neighbor)
                self.maxProfit=max(self.maxProfit,nextProfit)
            
    def maxProfit(self, prices):
        self.maxProfit=float("-inf")
        self.dfsR((-1,True,0,True),prices)
        return self.maxProfit
---------------------------------------------------------------------
ALGO: TREE_STYLE_DFS_HEAD_RECURSIVE WITH/WITHOUT MEMO

#ALGO_NAME: TREE_STYLE_DFS_HEAD_RECURSIVE
f() is the max profit you can get from an array starting at i
1. Recurrence relationship: f(i,1)=max(f(i+1,0)-prices[i],f(i+1,1))
                            f(i,0)=max(f(i+1,1)+prices[i],f(i+1,0))
                            
I added an extra state to record previous state: this will help to decide if I need to go for cooldown.
This question doesnt follow our pattern very cleanly. You can simply dfsR on all neighbors add to list, take a max or sum. Every neighbor has to be treated differently.

Base Cases are tricky to think about!!
Here the base case is i==n and NOT i=n-1 because there is a decision which can be taken at i=n-1 
BASE CASES HAPPEN WHEN ALL DECISIONS ARE TAKEN.
f(n,0)=0
f(n,1)=0. Why ? what is the maximum profit you can get from an empty list? 0


class Solution(object):
    logging=False
    def dfsR(self,root,prices):
        index,can_buy,prev_can_buy=root
        if index==len(prices):           ### base case    
            return 0
        if root in self.memo:
            return self.memo[root]
        
        if self.logging: print("root",index,can_buy,prev_can_buy)
        if can_buy and not prev_can_buy: ### cooldown
            a=self.dfsR((index+1,can_buy,can_buy),prices)
            if self.logging: print("cooldown",(index+1,can_buy,can_buy))
            ansRoot=a
        else:
            indicatorF=-1 if can_buy else 1
            if self.logging: print("change neighbor",(index+1,not(can_buy),can_buy))
            if self.logging: print("DN neighbor",(index+1,can_buy,can_buy))
            b=self.dfsR((index+1,can_buy,can_buy),prices)
            c=self.dfsR((index+1,not(can_buy),can_buy),prices)
            ansRoot=max(c+indicatorF*prices[index],b)
        
        self.memo[root]=ansRoot
        
        return self.memo[root]
            
    def maxProfit(self, prices):
        self.memo={}
        return self.dfsR((0,True,True),prices)

---------------------------------------------------------------------
Solution2 : BOTTOM_UP_TABULATION
My state space has 3 variables. How to deal with it. Is it possible to do this.
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/1981967/c%2B%2B-dp-with-memoization-and-tabulation-solution
Later
======================================================================
123. Best Time to Buy and Sell Stock III

Here its asking me that maximum two transactions are allowed. so we can only grow a tree which is valid in case of this condition.
When selling is done Each transaction is complete, not necessarily buying. so we increase counter when selling happens 
and try to add it as a STATE VARIABLE. 
When does selling happen when can_buy goes from False to True. 

Max profit check at every step or at leaves? At every step. NO!!!! LEAF IS ENOUGH!!
What about this point ? I dont need to check at every step because there is a different branch of tree that is including that possibility!!!

What to return at the base care when transaction =2 ? 
ALL profits have been added already and account has been updated. So its ok to return 0 similar to array finish case



class Solution(object):
    logging=True
    def dfsR(self,root,prices):
        index,can_buy,trans_count=root
        if index==len(prices):           ### base case    
            return 0
        if trans_count==2:               ### I was very confused so as to return what here
            return 0
        
        if root in self.memo:
            return self.memo[root]
        
        if self.logging: print("root",index,can_buy,trans_count)
        indicatorF=-1 if can_buy else 1
        
        if can_buy:
            if self.logging: print("buying neighbor",(index+1,not(can_buy),trans_count))
            if self.logging: print("DN neighbor",(index+1,can_buy,trans_count))
            b=self.dfsR((index+1,can_buy,trans_count),prices)
            c=self.dfsR((index+1,not(can_buy),trans_count),prices)
            ansRoot=max(c+indicatorF*prices[index],b)
        else:
            if self.logging: print("selling neighbor",(index+1,not(can_buy),trans_count))
            if self.logging: print("DN neighbor",(index+1,can_buy,trans_count))
            b=self.dfsR((index+1,can_buy,trans_count),prices)
            c=self.dfsR((index+1,not(can_buy),trans_count+1),prices)
            ansRoot=max(c+indicatorF*prices[index],b)
            
        
        self.memo[root]=ansRoot
        return self.memo[root]
            
    def maxProfit(self, prices):
        self.memo={}
        return self.dfsR((0,True,0),prices)
======================================================================
188. Best Time to Buy and Sell Stock IV
Same as before, we just need to add k instead of 2

class Solution(object):
    logging=False
    def dfsR(self,root,prices):
        index,can_buy,trans_count=root
        if index==len(prices):           ### base case    
            return 0
        if trans_count==self.k:               ### I was very confused so as to return what here
            return 0
        
        if root in self.memo:
            return self.memo[root]
        
        if self.logging: print("root",index,can_buy,trans_count)
        indicatorF=-1 if can_buy else 1
        
        if can_buy:
            if self.logging: print("buying neighbor",(index+1,not(can_buy),trans_count))
            if self.logging: print("DN neighbor",(index+1,can_buy,trans_count))
            b=self.dfsR((index+1,can_buy,trans_count),prices)
            c=self.dfsR((index+1,not(can_buy),trans_count),prices)
            ansRoot=max(c+indicatorF*prices[index],b)
        else:
            if self.logging: print("selling neighbor",(index+1,not(can_buy),trans_count))
            if self.logging: print("DN neighbor",(index+1,can_buy,trans_count))
            b=self.dfsR((index+1,can_buy,trans_count),prices)
            c=self.dfsR((index+1,not(can_buy),trans_count+1),prices)
            ansRoot=max(c+indicatorF*prices[index],b)
            
        
        self.memo[root]=ansRoot
        return self.memo[root]
            
    def maxProfit(self,k, prices):
        self.memo={}
        self.k=k
        return self.dfsR((0,True,0),prices)
======================================================================
714. Best Time to Buy and Sell Stock with Transaction Fee

## fee for each transaction so i simply add fee while selling
## NOT AT buying!! Is it ok ?
## Seems like the question is ok with either buying or selling? Yes 

class Solution(object):
    logging=False
    def dfsR(self,root,prices):
        index,can_buy=root
        if index==len(prices):           ### base case    
            return 0
        
        if root in self.memo:
            return self.memo[root]
        
        if self.logging: print("root",index,can_buy)
        indicatorF=-1 if can_buy else 1
        
        if can_buy:
            if self.logging: print("buying neighbor",(index+1,not(can_buy)))
            if self.logging: print("DN neighbor",(index+1,can_buy))
            b=self.dfsR((index+1,can_buy),prices)
            c=self.dfsR((index+1,not(can_buy)),prices)
            ansRoot=max(c+indicatorF*prices[index],b)
        else:
            if self.logging: print("selling neighbor",(index+1,not(can_buy)))
            if self.logging: print("DN neighbor",(index+1,can_buy))
            b=self.dfsR((index+1,can_buy),prices)
            c=self.dfsR((index+1,not(can_buy)),prices)
            ansRoot=max(c+indicatorF*prices[index]-self.fee,b)
            
        
        self.memo[root]=ansRoot
        return self.memo[root]
            
    def maxProfit(self, prices, fee):
        self.memo={}
        self.fee=fee
        return self.dfsR((0,True),prices)
======================================================================




======================================================================
494. Target Sum
You see there is a choice of +/- at every index. There is a Binary tree. For solving trees, we have three styles.
ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK
ALGO_NAME: TREE_STYLE_DFS_HEAD_RECURSIVE
ALGO_NAME: BOTTOM_UP_TABULATION
Lets draw the tree first and then decide.How does the TREE LOOK LIKE?
TREE STRUCTURE: 



ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK
---------------------------------------------
I didnt do much, standard algo. 
1. What is the state variable:I took  Remaining array and pathSum
2. Define Neighbors thats it 

class Solution(object):
    logging=False
    def dfs(self,nums,target):
        stack=[(nums,0)]  ##nums is the remaining value /unprocessed
        while stack:
            nums,pathSum=stack.pop()
            neighbors=[(nums[1:],pathSum+nums[0]),(nums[1:],pathSum-nums[0])]
            for neighbor in neighbors:
                nextNums,nextpathSum=neighbor
                if nextNums:   ### base case is empty list
                    stack.append(neighbor)
                elif nextpathSum==target:
                    self.count+=1
                    if self.logging: print("leaf node",neighbor)

    def findTargetSumWays(self, nums, target):
        self.count=0
        self.dfs(nums,target)
        return self.count
---------------------------------------------
ALGO_NAME: TREE_STYLE_DFS_HEAD_RECURSIVE
What is the recurrence relationship: 
Step0
Define f: f is the number of ways an array starting at i can achieve target k.

Step1 : Make a choice at 0
what is the ask? total number of ways. So we add. 
f(0,target)=f(1,target-nums[0]) + f(1,target+nums[0]) 
### THE ADDITION IS NOT TRIVIAL, WE ARE ADDING THE NUMBER OF WAYS FROM BOTH TREES
Step2: Generalize it
f(i,k)=f(i+1,k-nums[i])+f(i+1,k+nums[i])
Step3: Base Case
DONT TAKE BASE CASE TO BE THE FINAL INDEX, i==n-1, BASE CASE IS EMPTY LIST!!
Base Case, i=n ### EMPTY LIST
f(n,k). Now THIS IS VERY CONFUSING AND CRUCIAL!!!
++++++++++++++++++++++++++++++++++++++++++
f(n,k)=0 always but 1 when k=0, MEMORIZE THIS, An empty list can achieve only one target--> 0!
++++++++++++++++++++++++++++++++++++++++++


WHAT IS THE MISTAKE I AM MAKING WHILE CREATING RECURRENCE RELATIONSHIPS
===========================================================================
a) at any STAGE, we make a choice, THOSE CHOICES ARE NOT PART OF RECURRENCE RELATIONSHIP
b) THE CHOICE WE MAKE AT ANY STATE DOESNT IMPACT ITS OWN STATE!! IT IMPACTS THE NEXT STATE
a) DECISIONS ARE NOT TO BE INCLUDED in f's arguments. DECISIONS ARE NOT PART OF THE STATES!!!!!!
A DECISION IS SOMETHING WHICH TAKES ME FROM ONE STATE TO ANOTHER. ITS NOT A STATE VARIABLE. 
===========================================================================
Followed guidelines for TREE_STYLE_DFS_HEAD_RECURSIVE super accurately. resulting in this code. 

class Solution(object):
    logging=False
    def dfsR(self,root,nums,target):
        index,k=root
        if index==len(nums) and k==0:
            return 1
        elif index==len(nums) and k!=0:
            return 0
        
        if root in self.memo:
            return self.memo[root]
        
        neighbors=[(index+1,k-nums[index]),(index+1,k+nums[index])]
        if self.logging: print("=================================root",root)
        if self.logging: print("neighbors",neighbors)
        rootList=[]
        for neighbor in neighbors:
            nextIndex,nextK=neighbor
            neighbor_ans=self.dfsR(neighbor,nums,target)
            rootList.append(neighbor_ans)
        
        self.memo[root]=sum(rootList)
        return self.memo[root]
        
    def findTargetSumWays(self, nums, target):
        self.memo={}
        return self.dfsR((0,target),nums,target)
---------------------------------------------
ALGO_NAME: BOTTOM_UP_TABULATION

THIS IS A 0/1 KNAPSACK PROBLEM


Again we need the recurrence relation for this.
f(i,k)=f(i+1,k-nums[i])+f(i+1,k+nums[i])
Now the range of i is from 0 to n (not n-1)
and what is the range of k: -sum(nums) to sum(nums) in this problem
So I create a 2D DP table and update that. 

### One of the implementation issue I faced was that recurrence relation is between i & k. 
But my dp goes from -sum(nums) to sum(nums) 
So index 0 corresponds to target -sum(nums). So target to index --> add sum_nums

BOTTOM_UP_TABULATION_2D_BACKWARDS

class Solution(object):
    logging=True
    def findTargetSumWays(self, nums, target):
        dp=[[None for k in range(-1*sum(nums),sum(nums)+1)] for i in range(len(nums)+1)]
        ###ORIGINAL RANGES FOR TARGET ###BUT INDEX FOR ROWS
        
        #if self.logging: print(dp)
        ### filling bottom row of this table
        for k in range(-1*sum(nums),sum(nums)+1):   ###ORIGINAL RANGES FOR TARGET
            if k==0:
                dp[len(nums)][k+sum(nums)]=1        ### ADDING SUMNUMS EVERYWHERE TO GET TRUE INDEX
            else:
                dp[len(nums)][k+sum(nums)]=0
        
        ### I go up backwards in rows after filling the bottom row
        for i in range(len(nums)-1,-1,-1):
            for k in range(-1*sum(nums),sum(nums)+1,1):
                if k+nums[i]+sum(nums)<=len(dp[0])-1:   ### ADDING SUMNUMS EVERYWHERE TO GET TRUE INDEX
                                                        ### I HAVE TO CHECK IF THE INDEX EXCEEDS AFTER ADDING 
                                                        ### IN THAT CASE I RETURN 0 FOR THE INDEX EXCEED
                    dp[i][k+sum(nums)]=dp[i+1][k-nums[i]+sum(nums)]+dp[i+1][k+nums[i]+sum(nums)]
                else:
                    dp[i][k+sum(nums)]=dp[i+1][k-nums[i]+sum(nums)]
                #### RECURRENCE HOLDS AFTER THIS -- THIS IS A PRINCIPLE TO BE USED AGAIN
                

        if target+sum(nums)<=len(dp[0])-1 and target+sum(nums)>=0:
            return dp[0][target+sum(nums)] ### ADDING SUMNUMS EVERYWHERE TO GET TRUE INDEX AND IF IT DOESNT EXIST 
                                           ### RETURN 0
        else:
            return 0
----------------------------------------------------------------------
NOW GOING FORWARDS WHICH I DONT USE- matrix is just 

#f(n,k)=0 always but 1 when k=0, MEMORIZE THIS, An empty list can achieve only one target--> 0!
#f(i,k)=f(i+1,k-nums[i])+f(i+1,k+nums[i])
# Now the range of i is from 0 to n (not n-1)
# and what is the range of k:  -sum(nums) to sum(nums) not 0 to sum(nums)
class Solution(object):
    logging=False
    def findTargetSumWays(self, nums, target):
        dp = [[None for k in range(-1*sum(nums),sum(nums)+1)] for i in range(len(nums)+1)]
        #### Filled the 0th row ###0th row doesnt include 0th item
        for k in range(-1*sum(nums),sum(nums)+1):
            if k==0:
                dp[0][k+sum(nums)]=1 ###only one target can be achieved without any elements
            else:
                dp[0][k+sum(nums)]=0 
                
        for i in range(1,len(nums)+1):  ## i filled the 0th row so removing 
            for k in range(-1*sum(nums),sum(nums)+1): ## want to fill all k
                # if k+nums[i]+sum(nums)<=len(dp[0])-1:
                #     dp[i][k+sum(nums)]=dp[i-1][k+sum(nums)+nums[i-1]]+dp[i][k+sum(nums)-nums[i]]
                
                if k+sum(nums)+nums[i-1]<=len(dp[0])-1 and k+sum(nums)-nums[i-1]>=0:
                    dp[i][k+sum(nums)]=dp[i-1][k+sum(nums)+nums[i-1]]+dp[i-1][k+sum(nums)-nums[i-1]]
                elif k+sum(nums)+nums[i-1]>len(dp[0])-1 and k+sum(nums)-nums[i-1]>=0:
                    dp[i][k+sum(nums)]=dp[i-1][k+sum(nums)-nums[i-1]]
                else:
                    dp[i][k+sum(nums)]=dp[i-1][k+sum(nums)+nums[i-1]]
        #print(dp)            
        if target+sum(nums)>=0 and target+sum(nums)<=len(dp[0])-1:
            return dp[-1][target+sum(nums)]
        else:
            return 0        
======================================================================
139. Word Break
I can make a partition  or not partition at every index but this is not needed.
So I calculate a list of len of words in dicts and I make partitions at all every node as many as the lengths.
I only make neighbors when the left out list is a valid string. I made a graph to be super clear.
What is the state variable? I just need the remaining string

What is the base case ? String will be empty at last

TREE STRUCTURE: CHOOSE_ALL_POSSIBLE_CHILDREN



class Solution(object):
    logging=False
    def dfsR(self,root):
        if root=="":
            return True
        if root in self.memo:
            return self.memo[root]
        
        if self.logging: print("current",root)
        neighbors=[]
        for length in self.len_lists:
            if root[:length] in self.wordDict:
                neighbors.append(root[length:])
        if self.logging: print("neighbors",neighbors)  
        
        root_list=[]
        for neighbor in neighbors:
            neighbor_ans=self.dfsR(neighbor)
            root_list.append(neighbor_ans)
        #print("root_ans",[x==True for x in root_list])
        root_ans=any(root_list)
        self.memo[root]=root_ans
        return self.memo[root]

    def wordBreak(self, s, wordDict):
        self.len_lists = [len(word) for word in wordDict]
        ### dedupe this len_list to speeden up the code and remove duplicate states
        self.len_lists.sort()
        self.wordDict=collections.Counter(wordDict)
        self.memo={}
        return self.dfsR(s)    
======================================================================
140. Word Break II
I find it harder to come up with DFS_HEAD_RECURSIVE SOLUTION because the recurrence condn is difficult to figure out.
So I just went with the standard ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK
Followed pattern to the T. 

Had to do some duplicate removal work. 
First removed duplicates in len_list
Didnt add to neighbor if len exceeded len_rem_string. NOT len_rem_string-1 as i thought earlier.


class Solution(object):
    logging=False
    def dfs(self,s,wordDict):
        stack=[(s,[])]
        ret=[]
        while stack:
            rem_str,list1=stack.pop()
            
            if self.logging: print("current",rem_str,list1)
            neighbors=[]
            for length in self.len_lists:
                if rem_str[:length] in self.wordDict and length<=len(rem_str):
                    neighbors.append((rem_str[length:],list1+[rem_str[:length]]))
            
            #if self.logging: print("neighbors",neighbors)
            
            for neighbor in neighbors:
                nextStr,nextList1=neighbor
                if len(nextStr)>0:
                    stack.append(neighbor)      ## continuation
                    if self.logging: print("stack",stack)
                else:
                    if self.logging: print("leaf",nextList1)
                    ret.append(nextList1)       ## answer leaf node
        #print(ret)          
        return [" ".join(x) for x in ret]
                    

    def wordBreak(self, s, wordDict):
        self.len_lists=set([len(word) for word in wordDict])
        #print(self.len_lists)
        self.wordDict = collections.Counter(wordDict)
        return self.dfs(s,wordDict)
======================================================================




======================================================================
403. Frog Jump
ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK -- TLE
Mistakes : 
1. Tried to pass remaining list and at the same time use a map for indices
2. Didnt make sure jumps are not negative or 0


class Solution(object):
    logging=False
    def dfs(self,stones):
        stack=[(0,0)]
        while stack:
            index,prevJump=stack.pop()
            
            ##find neighbors
            neighbors=[]
            if stones[index]+prevJump in self.stoneMap and prevJump>=1:
                neighbors.append((self.stoneMap[stones[index]+prevJump],prevJump))
            if stones[index]+prevJump+1 in self.stoneMap and prevJump+1>=1:
                neighbors.append((self.stoneMap[stones[index]+prevJump+1],prevJump+1))
            if stones[index]+prevJump-1 in self.stoneMap and prevJump-1>=1:
                neighbors.append((self.stoneMap[stones[index]+prevJump-1],prevJump-1)) 
            if self.logging: print("stones",stones)
            if self.logging: print("===============================root",index,prevJump)
            if self.logging: print("neighbors",neighbors)
            
            for neighbor in neighbors:
                nextIndex,nextPrevJump=neighbor
                if nextIndex<len(stones)-1:
                    stack.append(neighbor)
                elif nextIndex==len(stones)-1:
                    if self.logging: print("leaf node",neighbor)
                    return True
        return False    
                
    def canCross(self, stones):
        self.stoneMap={}
        for i in range(1,len(stones),1):
            self.stoneMap[stones[i]]=i
        return self.dfs(stones)
------------------------------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_DFS_HEAD_RECURSIVE
What is the recurrence relationship?
f(i,k)= f(i+k,k)|| f(i+k+1,k+1)|| f(i+k-1||k-1)
What is the base case? No decision making,all decisions have been made i==n-1
f(n-1,k)=True for a valid k

class Solution(object):
    logging=False
    def dfsR(self,root,stones):
        index,prevJump=root
        if index==len(stones)-1:
            if self.logging: print("leaf node",index,prevJump)
            return True
        
        if root in self.memo:
            return self.memo[root]
    
        neighbors=[]
        if stones[index]+prevJump in self.stoneMap and prevJump>=1:
            neighbors.append((self.stoneMap[stones[index]+prevJump],prevJump))
        if stones[index]+prevJump+1 in self.stoneMap and prevJump+1>=1:
            neighbors.append((self.stoneMap[stones[index]+prevJump+1],prevJump+1))
        if stones[index]+prevJump-1 in self.stoneMap and prevJump-1>=1:
            neighbors.append((self.stoneMap[stones[index]+prevJump-1],prevJump-1)) 
        if self.logging: print("stones",stones)
        if self.logging: print("===============================root",index,prevJump)
        if self.logging: print("neighbors",neighbors)
        
        root_List=[]
        for neighbor in neighbors:
            nextIndex,nextPrevJump=neighbor ### DONT HAVE TO CHECK HERE AS WE DO IT ITERATIVE
            neighbor_ans=self.dfsR(neighbor,stones)
            root_List.append(neighbor_ans)
        if self.logging:print("root_list",root_List)
        
        self.memo[root]=any(root_List)  
        
        return self.memo[root] 
                
    def canCross(self, stones):
        self.memo={}
        self.stoneMap={}
        for i in range(1,len(stones),1):
            self.stoneMap[stones[i]]=i
        return self.dfsR((0,0),stones) 
------------------------------------------------------------------------------------------
ALGO_NAME: BOTTOM_UP_TABULATION -- Memory limit exceeded
f(i,k)= f(i+k,k)|| f(i+k+1,k+1)|| f(i+k-1||k-1)
Range of i is from 0,n-1 what about k ? can be from 1 to len(stones)
Tabulation is working conceptually but 0 <= stones[i] <= 231 - 1
So delta will be HUGE!!
class Solution(object):
    logging=False
    def canCross(self, stones):
        if stones[0]+1<stones[1]:
            return False
        
        delta=stones[-1]-stones[0]
        self.stoneMap={}
        for i in range(1,len(stones),1):
            self.stoneMap[stones[i]]=i
        dp=[[None for k in xrange(1,delta+1)] for i in range(len(stones))]
        #if self.logging: print(dp)
        for k in range(1,delta+1):   ###ORIGINAL RANGES 
            dp[len(stones)-1][k-1]=True        ### subtracting 1 EVERYWHERE TO GET TRUE INDEX

        for i in range(len(stones)-2,-1,-1):
            for k in range(1,delta+1):
                #if i==0: k=0
                if k-1>=1 and k-1<=delta and stones[i]+k in self.stoneMap and  self.stoneMap[stones[i]+k]>=0 and self.stoneMap[stones[i]+k]<=len(stones)-1:
                    a=dp[self.stoneMap[stones[i]+k]][k-1]
                else:
                    a=False
                if k>=1 and k<=delta and stones[i]+k+1 in self.stoneMap and self.stoneMap[stones[i]+k+1]>=0 and self.stoneMap[stones[i]+k+1]<=len(stones)-1:
                    b=dp[self.stoneMap[stones[i]+k+1]][k+1-1]
                else:
                    b=False
                if k-2>=1 and k-2<=delta and stones[i]+k-1 in self.stoneMap and self.stoneMap[stones[i]+k-1]>=0 and self.stoneMap[stones[i]+k-1]<=len(stones)-1:
                    c=dp[self.stoneMap[stones[i]+k-1]][k-1-1]
                else:
                    c=False
                
                
                dp[i][k-1]= a or b or c
                #### RECURRENCE HOLDS AFTER THIS -- THIS IS A PRINCIPLE TO BE USED AGAIN
        print(dp)

        if True:
            return dp[1][0]### ADDING SUMNUMS EVERYWHERE TO GET TRUE INDEX AND IF IT DOESNT EXIST 
                                           ### RETURN 0
        else:
            return False
------------------------------------------------------------------------------------------
Now this dp can be optimized instead of keeping all True & False, we can only store True.
We create a map corresponding to each row,  





======================================================================








++++++++++++++++++++++++++++++++
+++++++Group : BACKTRACKING +++
++++++++++++++++++++++++++++++++ 
Python backtrack solutions
https://leetcode.com/problems/subsets/discuss/429534/General-Backtracking-questions-solutions-in-Python-for-reference-%3A


46. Permutations
Now this question used to scare me earlier. But isnt this simple head recursive? LOL yes it is.
What is the structure of this recursion?
f(n)-> f(n-1)-> f(n-2)->f(n-3)->f(1)->f(0) found answer at f(0) then bubble it up.
YOU ARENT USING THE CLASSIC TREE STRUCTURES IN OTHER SOLUTIONS HERE! YOUR TREE STRUCTURE IS LINEAR
[[3]]

[[23],[32]]

[[123],[213],[231],  [132],[312],[321]] ## 6 answers

class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if not nums:
            return [[]]     ## its important to write like this not [] because the function return type is list of list
        
#         if len(nums)==1:  ##either base case is fine 
#             return [nums]   
        list1=self.permute(nums[1:])
        #print(list1)
        list2=[]
        for i in range(len(list1)):
            for j in range(len(list1[i])):
                    list2.append(list1[i][:j]+[nums[0]]+list1[i][j:])       ## for insert the return type is null
            list2.append(list1[i]+[nums[0]])    ##dont forget the list around nums
        return list2

Time Complexity: Time: O(N*N!) why?
---------------------------------------------------------------------
Now I can create a bloody tree too and iterate that tree. whats the tree?
                   [],nums
            [1]       2       3
          12,13   21,23    31,32
         123,132 213,231  312,321       ANS ## 6 leaf nodes
         
Lets iterate this tree ? how DFS 
But we realize there is an additions list at each state(nums)
https://leetcode.com/problems/permutations/discuss/993970/Python-4-Approaches-%3A-Visuals-%2B-Time-Complexity-Analysis
I try to follow approach : Approach 3 : DFS Iterative with Explicit Stack

ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
1. We dont have visited because its a TREE
2. 

### dfs ## pretty straighforward
class Solution(object):
    def dfs(self,nums):
        ans=[]
        stack=[([],nums)]
        while stack:
            path,nums=stack.pop()
            neighbors=[(path+[nums[i]],nums[:i]+nums[i+1:]) for i in range(len(nums))]
            for neighbor in neighbors:
                nextPath,nextnums=neighbor
                if nextnums:                    ### condition condition will be on next variable
                    stack.append(neighbor)
                else:                           ### This is the leaf node or Ans node, we dont let it enter stack
                    ans.append(nextPath)        ### we return nextPath variable
        return ans
    

    def permute(self, nums):
        return self.dfs(nums)
They call this backtracking?? This is tree traversal
We build the graph here while,we are given the graph in trees/graphs

Now what is the difference between top-down memo approaches and this dfs traversal approach.
In top-down memo approach too, no calculation is done while coming from root to leaf, at leaf base case hits and then while going up all calculations are done. top-down memo approach is same as a HEAD recursion but with a dictionary added so that we dont repeat work.

While this is closer to tail recursion, because we do the work while going down and once we hit the leaf. We find the answer.
======================================================================
TREE                        DP STYLE QUESTIONS
HEAD RECURSION              TOP-DOWN WITH MEMO  (We start at the top do nothing, hit the base and then start doing the work while going up)        
TAIL RECURSION              DFS/BFS STYLE TRAVERSAL WHICH DOES THE WORK WHILE COMING DOWN(ALSO CALLED BACKTRACK)
                            BOTTOM_UP WITH TABULATION (here we have access to leaves somehow and we start from leaves to root)

Recurrence relationship is helpful in head, its corollary(topdownmemo) and bottomUpTabulation
======================================================================
47. Permutations II

Its just removing dups from Permutations I

class Solution(object):
    def permuteUnique(self, nums):
        if not nums:
            return [[]]
        
        list1=self.permuteUnique(nums[1:])
        list2=[]
        for i in list1:
            for j in range(len(i)):
                if i[:j]+[nums[0]]+i[j:] not in list2:
                    list2.append(i[:j]+[nums[0]]+i[j:])
            if i+[nums[0]] not in list2:
                list2.append(i+[nums[0]])
                
        return list2
Can use a hash to optimise
---------------------------------------------------------------------
DFS/BFS STYLE TRAVERSAL/BACKTRACKING
ALGO_NAME: TREE_STYLE_DFS_BACKTRACK

class Solution(object):
    def dfs(self,nums):
        ans=[]
        stack=[([],nums)]
        while stack:
            path,nums=stack.pop()
            neighbors=[(path+[nums[i]],nums[:i]+nums[i+1:]) for i in range(len(nums)) if i==0 or nums[i]!=nums[i-1]] ### SAFROT RULE
            for neighbor in neighbors:
                nextPath,nextnums=neighbor
                if nextnums:                ## continuation condn on next
                    stack.append(neighbor)
                else:
                    ans.append(nextPath)    ## return next as answer
        return ans
    
    
    def permuteUnique(self, nums):
        nums.sort()     ## I sort it which brings the duplicates next to each other
        return self.dfs(nums)  
======================================================================
78. Subsets
Again simple head recursion is effective 
f(n)-> f(n-1)-> f(n-2)->f(n-3)->f(1)->f(0) found answer at f(0) then bubble it up.
## See using which rule you can go from 2 to 3
#  [[],[2],[3],[2,3]] n=2
# [],[1],[2],[2,1],[3],[3,1],[2,3],[2,3,1] n=3 

ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
class Solution(object):
    def subsets(self, nums):
        if not nums:
            return [[]]
        
        list1=self.subsets(nums[1:])
        list2=[]
        for i in range(len(list1)):
            list2.append(list1[i])
            list2.append(list1[i]+[nums[0]])
        return list2
---------------------------------------------------------------------
# [][1,2,3]
# [1][2,3] [2][3]  [3][] [][]

DFS/BFS STYLE TRAVERSAL/BACKTRACKING
class Solution(object):
    def dfs(self,nums):
        ans=[]
        stack=[([],nums)]
        while stack:
            path,nums=stack.pop()
            neighbors=[(path+[nums[i]],nums[i+1:]) for i in range(len(nums))]+[(path,[])] ###
            for neighbor in neighbors:
                nextPath,nextnums=neighbor
                if nextnums:                ###  continue traversal condn
                    stack.append(neighbor)
                else:
                    ans.append(nextPath)    ## TREE TRAVERSAL IS OVER ONLY WHEN NUMS IS NOT THERE, answer isnt added to stack either
        
        return ans

    def subsets(self, nums):
        return self.dfs(nums) 

HERE I ADD THE SAME PATH AGAIN AS A NEW LEAF AND SET LIST TO EMPTY, we could improve that by returning answer there only without considering it a 
neighbor and create a leaf
======================================================================
90. Subsets II
## HEAD RECURSION SOLUTION
# [][2]
#  [[],[2],[2],[2,2]]
# [],[1],[2],[2,1],[2],[2,1],[2,2],[2,2,1]
to avoid dups i did two things, check if its in list2 from before and only add sorted stuff
class Solution(object):
    def subsetsWithDup(self, nums):
        if not nums:
            return [[]]
        
        list1=self.subsetsWithDup(nums[1:])
        list2=[]
        for i in range(len(list1)):
            if sorted(list1[i]) not in list2:
                list2.append(sorted(list1[i]))
            if sorted(list1[i]+[nums[0]]) not in list2:
                list2.append(sorted(list1[i]+[nums[0]]))
        return list2
---------------------------------------------------------------------
# [][1,2,2]
# [1][2,2] [2][2]  [2][] 

# [][1,2,2]
# [1][2,2] [2][2]  [2][] 
DFS/BFS STYLE TRAVERSAL/BACKTRACKING, ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
class Solution(object):
    def dfs(self,nums):
        ans=[]
        stack=[([],nums)]
        while stack:
            path,nums=stack.pop()
            neighbors=[(path+[nums[i]],nums[i+1:]) for i in range(len(nums)) if i==0 or nums[i]!=nums[i-1]] +[(path,[])] ###
            for neighbor in neighbors:
                nextPath,nextnums=neighbor
                if nextnums:                ###  continue traversal condn 
                    ### tree violation check happens before addition to stack ### invalid cases arent added to stack
                    stack.append(neighbor)
                else:                   ## TREE TRAVERSAL IS OVER ONLY WHEN NUMS IS NOT THERE #answer isnt added to stack either
                    ans.append(nextPath)  
            
        return ans

    def subsetsWithDup(self, nums):
        nums.sort()
        return self.dfs(nums)  
        
Note:
=====================================================
permutation its path+nums[i] and nums[:i]+nums[i+1:]
subsetting its path+nums[i] and nums[i+1:]
====================================================
======================================================================
77. Combinations

DFS/BFS STYLE TRAVERSAL/BACKTRACKING, ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
# ## n=4, k=2 [1,2,3,4]
# [],i=0
# 1,i=1       2,i=2       3,i=3       4,i=4
# 12,13,14    23,24       34  

class Solution(object):
    def dfs(self,n,k):
        ans=[]
        stack=[([],0)]
        while stack:
            path,index=stack.pop()
            neighbors=[(path+[i+1],i+1) for i in range(index,n,1)] 
            for neighbor in neighbors:
                nextPath,nextIndex=neighbor
                if len(nextPath)<k:  ## recursion violation checked before adding
                    stack.append(neighbor)
                else:
                    ans.append(nextPath) ## answer only in case of stopping #answer isnt added to stack either
        return ans
    
    def combine(self, n, k):
        return self.dfs(n,k)
======================================================================
216. Combination Sum III (THREE)
# []
# 1 []  ......2 3 4 
# 12 ...
# 123 .....
This question is exactly like SUBSETS, but we have an additional conditions to care about, len and sum

ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
## use more deault formats 
class Solution(object):
    def combinationSum3(self,k,n):
        nums=range(1,10)
        stack=[([],nums,0)] ### ALWAYS IN TUPLE INSIDE A STACK ## RULE
        ans=[]
        while stack:
            path,nums,pathSum=stack.pop()
            #print(path,nums)
            neighbors=[(path+[nums[i]],nums[i+1:],pathSum+nums[i]) for i in range(len(nums))]
            #print(neighbors)
            #print("==================")
            for neighbor in neighbors:
                nextPath,nextNums,nextPathSum=neighbor
                if nextNums and len(nextPath)<k:    ## nextNums is not really needed because the other conditions fails faster but due to synatx conform
                    stack.append(neighbor)
                elif len(nextPath)==k and nextPathSum==n:   ### if i dont put things in stack the traversal stops automatically
                    ans.append(nextPath)
            
        return ans
======================================================================
40. Combination Sum II (SIMILAR TO SUBSET2 WITH DUPLICATES)
# []
# 1,[2,3,4] 2,[3,4]
ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
class Solution(object):
    def dfs(self,nums,target):
        ans=[]
        stack=[([],nums,0)]
        while stack:
            path,nums,pathSum=stack.pop()
            neighbors=[(path+[nums[i]],nums[i+1:],pathSum+nums[i]) for i in range(len(nums)) if i==0 or nums[i]!=nums[i-1]]
            for neighbor in neighbors:
                nextPath,nextNums,nextPathSum=neighbor
                if nextNums and nextPathSum<target:        ### tree violation check happens before addition to stack
                    stack.append(neighbor)
                elif nextPathSum==target:                  ### answer found condition  ## answer isnt added to stack either
                    ans.append(nextPath)                       
        
        return ans
    
    def combinationSum2(self, candidates, target):
        candidates.sort()
        return self.dfs(candidates,target)
======================================================================
39. Combination Sum
Dups is the main issue
Solution1)  I added sorted lists to the ans and check before adding new things 
ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
class Solution(object):
    def dfs(self,nums,target):
        ans=[]
        stack=[([],0)]
        while stack:
            path,pathSum=stack.pop()
            neighbors=[(path+[nums[i]],pathSum+nums[i]) for i in range(len(nums))]
            for neighbor in neighbors:
                nextPath,nextPathSum=neighbor
                if nextPathSum<target:                                    ###continue traversal condn
                    stack.append(neighbor)
                elif nextPathSum==target and sorted(nextPath) not in ans: ###answer found condition #answer isnt added to stack either
                    ans.append(sorted(nextPath))   
        return ans 
    
    def combinationSum(self, candidates, target):
        return self.dfs(candidates,target)
Solution2)  I added one thing and dont add it for sure in the other branch. Why does this work? 

======================================================================
22. Generate Parentheses
going from n-1 to n didnt work.

here we have only two neighbors.We maintain two counts.
focus on stopping conditions.
we can continue if count1>=count2 or if we havent exhausted 2n
answer found when count1 n and count2 n 
LGO_NAME: TREE_STYLE_DFS_BACKTRACK
class Solution(object):
    def dfs(self,n):
        ans=[]
        stack=[([],0,0)]
        while stack:
            path,count1,count2 = stack.pop()
            neighbors=[(path+["("],count1+1,count2),(path+[")"],count1,count2+1)]
            for neighbor in neighbors:
                nextPath,nextCount1,nextCount2=neighbor
                if nextCount1>=nextCount2 and nextCount1+nextCount2<2*n: ### continue condn
                    stack.append(neighbor)
                elif nextCount1==n and nextCount2==n:                    ### answer found condn ## answer isnt added to stack either
                    ans.append("".join(nextPath))
        return ans
    def generateParenthesis(self, n):
        return self.dfs(n)
======================================================================
17. Letter Combinations of a Phone Number
Standard backtrack

# []"23" "abc" 
# [a]"3" [b]"3" [c]"3"
ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
class Solution(object):
    def dfs(self,digits):
        
        ans=[]
        dict1={'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        stack=[([],digits)]
        while stack:
            path,digits=stack.pop()
            neighbors=[(path+[dict1[digits[0]][i]],digits[1:]) for i in range(len(dict1[digits[0]]))]
            for neighbor in neighbors:
                nextPath,nextDigits=neighbor
                if nextDigits:                     ### traversal continue condn
                    stack.append(neighbor)
                else:                              ### answer found condition  ## answer isnt added to stack either
                    ans.append(''.join(nextPath))  
        return ans
    
    
    def letterCombinations(self, digits):
        if not digits:      ### edge case needs to be handled, see not works in case of empty string, list, dict1
            return []
        return self.dfs(digits)
======================================================================
1980. Find Unique Binary String
Basic BFS traversal, just checking at lenth n
ALGO_NAME: TREE_STYLE_BFS_BACKTRACK
class Solution(object):
    def dfs(self,nums):
        dict1=collections.Counter(nums)
        ans=[]
        stack=[[]]
        while stack:
            path=stack.pop()
            neighbors=[path+["0"],path+["1"]]
            for neighbor in neighbors:
                nextPath=neighbor
                if len(nextPath)<len(nums):                     ### traversal continue condn
                    stack.append(neighbor)
                elif ''.join(nextPath) not in dict1:     ### answer found condition  ## answer isnt added to stack
                    return ''.join(nextPath)
    
    def findDifferentBinaryString(self, nums):
        return self.dfs(nums)
======================================================================
79. Word Search
This question killed me in iterative :( :(. What is the issue? The issue is the visited dictionary. This problem came as a surprise.
We maintain a visited dictionary for each dfs loop, now when we backtrack and  revisit the same node, we cant because we marked it as visited.

If we dont use visited at all ,nothing stops me from going back, so what do i need. I need a visited which just revert when the stack returns.
So i need to keep visited as a part of the STATE variable in dfs stack.
https://leetcode.com/problems/word-search/discuss/1811014/Iterative-and-Recursive-DFS-Solution-in-Python

## TLE -- Iterative
ALGO_NAME: TREE_STYLE_DFS_BACKTRACK.
visited set is a part of the stack
class Solution(object):
    def dfs(self,i,j,board,word):
        m=len(board)
        n=len(board[0])
        dirs=[[1,0],[-1,0],[0,1],[0,-1]]
        stack=[(i,j,word,set((i,j)))]
        while stack:
            currentX,currentY,word,visited=stack.pop()
            #print((currentX,currentY),board[currentX][currentY],self.word1,word)
            neighbors=[]
            for dir1 in dirs:
                d=visited.copy()
                d.add((currentX+dir1[0],currentY+dir1[1]))
                neighbors.append((currentX+dir1[0],currentY+dir1[1],word[1:],d))
                
            for neighbor in neighbors:
                nextX,nextY,word,nextvisited=neighbor
                
                if word and nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<= n-1 and board[nextX][nextY]==word[0] and (nextX,nextY) not in visited and neighbor not in stack:
                    stack.append(neighbor)
                    #visited[(nextX,nextY)]=1
                elif not word:                        ## answer found condition
                    self.found1=True
                    return 
    def exist(self, board, word):
        self.word1=word
        m=len(board)
        n=len(board[0])
        if m*n<len(word):
            return False
        self.found1=False
        for i in range(m):
            for j in range(n):
                if board[i][j]==word[0]:
                    if not self.found1:
                        self.dfs(i,j,board,word)
                    else:
                        return True
        return self.found1
        
Will have to look for recursive solution for this because recursion has natural backtrack        
======================================================================
ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
51. N-Queens
Hard : Took 1.5 hrs to code after knowing the solution. 
0. This is obviously a backtracking question why? so many possibilities.
1. We increase row one by one, so row cant coincide, we maintain a dict for column, for +ve diag, -ve diag
2. +ve diagonal is one with positive slope, i-j is the dict key, -ve diagonal, i+j is dict key
3. Now the CRUCIAL PART IS How can we define a STATE and what are the vars inside the tuple of state. Visualize the tree. When we move from A to B and A to C. What cant be common in B and C.  
4. Here the all 3 dicts cant be common. Have to include them in state space. Here we also need a list of coordinates. Its best to add row too to the state space. Why ? This will help us to go from A to B. 
Now its easy and just the structure of Backtracking iteration algorithm needs to be remembered.

class Solution(object):
    def dfs(self,n):
        logging=True                        ### USE LOGGING
        stack=[([],0,set(),set(),set())]    #### FIRST POSITION STACK IS ALWAYS HARDCODED, NOTICE THE SETS ## NEVER USE DICTS
        ans=[]
        while stack:
            list_cords,i,column,positiveDiag,negativeDiag=stack.pop() ### STACK POP WILL ALWAYS HAVE VARIABLE NAMES. 
            #if logging: print("list_cords",list_cords)
            neighbors=[]                                              ### IN DEFINITION OF NEIGHBORS WE USE THE VARS FROM STACK POP!!!!!
                                                                      ### WE COPY THE SETS FIRST AND THEN ADD !!!!
                                                                      ### FOR LISTS TOO WE TAKE CARE NOT TO ADD THE REFERENCE. 
            for j in range(n):
                if j not in column and i-j not in positiveDiag and i+j not in negativeDiag: 
                    a=column.copy()
                    a.add(j)
                    b=positiveDiag.copy()
                    b.add(i-j)
                    c=negativeDiag.copy()
                    c.add(i+j)
                    neighbors.append((list_cords+[(i,j)],i+1,a,b,c))
            if logging: print("neighbors",neighbors)
            for neighbor in neighbors:
                nlist_cords,ni,ncolumn,npositiveDiag,nnegativeDiag=neighbor ### WE MUST GET NEW VARIABLES FOR THE NEIGHBOR
                ni,nj=nlist_cords[-1]
                #if logging: print("neighbor",neighbor)
                if ni!=n-1:                                                  ### I WRITE continuation condn using NEIGHBOR VARS!!!
                    stack.append(neighbor)
                    #if logging: print("stack",stack)
                elif ni==n-1:                                                ### I WRITE ANSWER CONDN, I DONT ADD ANSWER TO STACK
                    if logging: print("nlist_cords",nlist_cords)
                    ans.append(nlist_cords)                                 ### I APPEND THE NEIGHBOR PATH, NOT THE POPPED ONE
                    
            
        return ans 
                    
    def solveNQueens(self, n):
        ans=self.dfs(n)
        ans2=[]
        
        for list_cords in ans:
            grid=[["." for j in range(n)] for i in range(n)]        ## HAD TO regenerate grid
            for cords in list_cords:
                grid[cords[0]][cords[1]]="Q"        
            ans2.append(["".join(row) for row in grid])             ## remember how to get rows. we did it before will use it again 
        return ans2
======================================================================  
416. Partition Equal Subset Sum
1. Subset word tells me we have a TREE- So now brute force DFS (backtracking),DP or memo.  
1. We know how to calculate Subsets. Brute Force just use that and see if any subset sum is sum/2. 

ALGO_NAME: TREE_STYLE_DFS_BACKTRACK
class Solution(object):
    def dfs(self,nums):
        ans=[]
        if sum(nums)%2!=0: return False
        target=sum(nums)/2          ## be careful of integer division 
        stack=[([],nums,0)]
        while stack:
            path,nums,pathSum=stack.pop()
            #print(path,nums,pathSum)
            neighbors=[(path+[nums[i]],nums[i+1:],pathSum+nums[i]) for i in range(len(nums))]+[(path,[],pathSum)]
            
            for neighbor in neighbors:
                nextPath,nextnums,nextpathSum=neighbor
                if nextnums:                                 ## continuation condn 
                    stack.append(neighbor)
                elif nextpathSum==target:                   ## answer condn 
                    return True                   

        return False
                
    def canPartition(self, nums):
        return self.dfs(nums)
-------------------------------------------------------------
Just got all the sums?? lol what is this ?

class Solution(object):  
    logging=False
    def canPartition(self, nums):
        if sum(nums)%2!=0: return False
        sumSet=set()
        sumSet.add(0)
        for i in range(len(nums)-1,-1,-1):
            for sum1 in sumSet.copy():      ###.copy is essential while iterating on a set
                sumSet.add(sum1+nums[i])
                sumSet.add(nums[i])
        if sum(nums)/2 not in sumSet:
            return False
        else:
            return True
-------------------------------------------------------------   
Ok i see the bloody tree, How do I convert this to DP
STEPS FOR DP OFFICIAL
Step 1. Make a choice on 1st element: Can be chosen or not chosen
Step 2. Write it for 0: I(0,target)=I(1,target-nums[0]) || I(1,target)
Step 3. Base Case: I(n-1,nums[n-1])=1  or I(n-1,0)=1 else I(n-1, others)=0
Step 4. Generalize it : I(i,sum1)=I(i+1,sum1-nums[i])||I(i+1,sum1)
Step 5. Try to start from the back, I(n-2,)
The problem is what is the target I want to check for? Populate it for all.
Whats all here?0 to sum(nums)? Whats in between? all integers ? 
GODDAMN the table will be so huge. But this how to solve it.
Step 6. Simplifying the dp array is possible if you realize there is only one previous array required.
Step 7. Further simplification is possible if instead of storing False, we just True values from the previous row 
and find the True values for the next row. 
THAT IS THE ABOVE SOLUTION!!!!!!!!!!!!!!!!!!!

ALGO_NAME:  BOTTOM_UP_DP_TABULATION  
## I(i,sum1)=I(i+1,sum1-nums[i])||I(i+1,sum1)
class Solution(object):  

    def canPartition(self, nums):
        target=sum(nums)/2
        if sum(nums)%2!=0: return False
        
        dp=[False for i in range(0,target+1,1)]
        dp[0]=True
        if nums[len(nums)-1]<=len(dp)-1:
            dp[nums[len(nums)-1]]=True
        for i in range(len(nums)-2,-1,-1):
            newdp=[False]*len(dp)
            for j in range(len(dp)):
                if dp[j]:
                    newdp[j]=True
                    if j+nums[i]<=len(dp)-1:
                        newdp[j+nums[i]]=True
            dp=newdp
        #print(dp)
        return dp[target] ### we want to check if a particular target exists.
======================================================================
351. Android Unlock Patterns 
We have a tree starting at every point. 
Now I spent an hour on DP solution. But DP wont work because we have to keep track of visited. and we cant go to the same point once again. 
Okay 



======================================================================  
        
======================================================================
++++++++++++++++++++++++++++++++
+++++++Group : GREEDY +++
++++++++++++++++++++++++++++++++
134. Gas Station
Brute Force Solution in O(N**2)
Check every point on the array, and go forward and check the whole loop. 


SUPER UNINTUITIVE GREEDY O(n) solution
https://www.youtube.com/watch?v=lJwbPZGo05A&list=PLbfzAGJjuwAL31__uSFXBK26D9q4Ythb0&index=18
0. Make sure sum(gas)>=sum(cost)
1. Start at index 0 and move forward till gas doesnt run out. 
2. Start again from the next point where gas ran out. Ignore all the points in between because they arent possible answers. Why?
If we include the starting point(which had positive contribution, we didnt reach we definitely wont reach if we exlude it). Repeat this 
3. If there exists a pount from where we reach end of array with positive gas, thats our answer. Why? Any points after this point have less totals again because the starting point had positive contribution. So if there is any possible point it is this 
4. Code wise just a single loop and remembering where we started from 

class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        if sum(gas)<sum(cost):
            return -1
        i=0
        total=0
        base=0
        while i<=len(gas)-1:
            total+=gas[i]-cost[i]
            #print(i,total)
            if total<0:         ##violation 
                total=0         ##correction
                base=i+1
            i+=1
        
        if total>=0:            ## equals zero also works 
            return base
======================================================================
2038. Remove Colored Pieces if Both Neighbors are the Same Color

# 1. Brute Force
# colors = "AAABABB"
# 1. no edges allowed 
# 2. surrounding elements same 
# 3. A for alice, b for bob
# Unable to move then you lose
# True if Alice wins and false when bob wins

# "AAABABB" 
# "AABABB"


# ABBBBBBBAAA
# ABBBBBBBAA

# Alice -- check if she can remove
# Lets say she can remove many, should we do greedy (first here)?
# or eval all options. Check will take orderN time

# Backtrack 

# 2. Optimised approach: 
# Greedy approach?
    
### crucial though process here -- thought myself
# only if Group of 3 or more -- removal possible 
# If group >=3, remove any in the middle is the same
# Lets say we have multiple groups? Will order matter? No 
# Can my groups merge because of other player moves? No because unless its a group of 3 Alice cant remove.

A group of k will produce k-2 moves. So i count groups of k >=3 and maintain counters. 

# ABBBBBBBAAA

# groups of A and B >=3 and sizes 
# A=[3] - 1
# B=[7] - 5

# 3. Edge cases 
# Didnt check 
# 4. Syntax

class Solution(object):
    def winnerOfGame(self, colors):
        """
        :type colors: str
        :rtype: bool
        """
        removalA=0
        removalB=0
        
        i=0
        countA=0
        countB=0
        while i<=len(colors)-1:   #ABBBBBBBAAA
            if colors[i]=="A":
                countA+=1
                if countB>=3:
                    removalB+=countB-2
                countB=0
            if colors[i]=="B":
                countB+=1
                if countA>=3:
                    removalA+=countA-2
                countA=0
            i+=1
        
        if countA>=3:               ## missed this edge condition in the first run
            removalA+=countA-2
        if countB>=3:
            removalB+=countB-2
    
        
        if removalA>removalB:       ### HAS TO BE GREATER BECAUSE ALICE MOVES FIRST
            return True
        else:
            return False  
======================================================================




======================================================================
31. Next Permutation

Brute force method will be to create the entire permutation tree
We define  Our anchor as the element which breaks the non-decreasing sequence( from left )
Once we find this element we know we need to swap it with the next biggest number. Because at this point we have exhausted all combinations starting with this number. So we simply take the next number. 
After swapping the resulting array is also strictly non-decreasing (why? because we put the swap in a place where it won't disturb the order). Now we just change decreasing to increasing by reversing. OF course the first element has to be increasing order(lexicographically first).
In case the loop doesnt break. All non decreasing. We have finished the permutation list. Now return the first eleemnt by just reversing.  (or SORTING! same thing)





++++++++++++++++++++++++++++++++
+++++++Group : STRINGS +++
++++++++++++++++++++++++++++++++   
======================================================================  
293. Flip Game
------------------------------------------------------------------------------------------
Done,#### STRINGS AND LISTS DO NOT GIVE INDEX ERROR(EMPTY) WHEN RANGE IS THERE OTHERWISE THEY WILL GIVE ERROR
------------------------------------------------------------------------------------------
class Solution(object):
    def generatePossibleNextMoves(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res=[]
        for i in range(len(s)):
            if s[i:i+2]=='++':
                res.append(s[:i]+'--'+s[i+2:])
            
        return res
======================================================================

383. Ransom Note
Tag: String 
------------------------------------------------------------------------------------------
Done
------------------------------------------------------------------------------------------
======================================================================
Logic : Just use a dict nothing much, very easy question 
---------------------------------------------------------------------
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        
        ## logic ok
        ## syntax ok
        ## edge 
        # ransomNote empty True True        
        # 
        dict1={}
        for i in range(len(magazine)):
            if magazine[i] not in dict1:
                dict1[magazine[i]]=1
            else:
                dict1[magazine[i]]+=1
        
        for i in range(len(ransomNote)):
            if ransomNote[i] not in dict1:
                return False
            else:
                if dict1[ransomNote[i]]>1:
                    dict1[ransomNote[i]]-=1
                else:
                    del dict1[ransomNote[i]]
        else:
            return True
======================================================================            
                    
        
++++++++++++++++++++++++++++++++++++++
+++++++Group : MATH +++
++++++++++++++++++++++++++++++++++++++
======================================================================
202. Happy Number

Logic: Very similar to the question where we find loops in a link list . Initiate fast and slow two ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“pointersÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢. While forever, increase fast two time and increase slow. Each time after increase fast and slow check if they are one or if they are equal or else continue the loop
 or we can use a dict like here

class Solution(object):
    def isHappy(self, n):
        dict1={}
        n=str(n)
        dict1[n]=1
        
        while True:
            sum1=0
            for i in range(len(n)):
                sum1+=int(n[i])**2
            n=str(sum1)
            if n=="1":
                return True
            elif n in dict1:
                return False
            dict1[n]=1
---------------------------------------------------------------------
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        str1=str(n)
        
        slow=str1
        fast=str1
        print fast
        
        while True:
            ans=0
            for i in range(len(fast)):
                ans=ans+int(fast[i])**2
            fast=str(ans)
            
            ans=0
            for j in range(len(fast)):
                ans=ans+int(fast[j])**2
            fast=str(ans)

            ans=0
            for k in range(len(slow)):
                ans=ans+int(slow[k])**2
            slow=str(ans)
            
            if slow=='1' or fast=='1':
                return True
            elif slow==fast:
                return False
            else:
                pass

======================================================================            
528. Random Pick with Weight
---------------------------------------------------------------------
Logic:

1. Essentially picking up numbers with a given probability.

How to achieve this, we create a cumulative sum and then random pick from 1 to end of cumsum.
[1,4,2] is array
[1,5,7] is cumsum
Now if you pick 1 ## go pick 1          1/7
Now if you pick 2,3,4,5 ## go pick 4    4/7
Now if you pick 6,7 ## go pick 2        2/7
which is our solution
But the question is asking for INDEX not number so once you narrow down left right, return right 

---------------------------------------------------------------------
So lets relate this problem with a problem of dividing a cake in a party such that the person with highest weight has better chance to pick a slice.(proportional to his weight)

Suppose we have people with weight 1, 3, 5, 7, 9 pounds and they went for a party to a bakery and purchased a cake. They decided that lets add our total weight and buy a cake proportional to it. So their totalWeight came as
1+3+5+7+9 = 25
They purchased a 25 pound cake :).
They decided that lets cut cake in 1 pound slices(25 slices total) and whoever wants to eat can take a 1 pound slice at a time. The person who will pick a slice will be picked randomly.

To find out how to pick randomly and to figure out who will pick first, they first did a cumulative sum of their weights

1->1
3-> 1 + 3 = 4
5-> 4 + 5 = 9
7-> 7 + 9 = 16
9-> 9 + 16 = 25

=1,4,9,16,25

And then asked the server to pick a random number out of 25 for them. The random number represents a slice.
So it can be 17 out of 25 or 6 out of 25.

So the slice 1 belongs to 1 pounds person. Which is only 1 slice.
Slice between 2-4 belong to 3 pounds person. Which are 3 slices.
.
.
Slice between 17- 25 belong to the 9 pounds person. Which are 9 slices.

If we pick a random slice out of 25, there is a higher probability that it will belong to a person with 9 slices (the person with 9 pounds) , the person who own 7 slices has second highest probability. The person whose weight is 1 pound and has only 1 slice has least probability to pick a slice.

And that's what we wanted. We want to let the person with highest weight get a greater chance to pick a slice every time even though they are picked at random.
---------------------------------------------------------------------
class Solution:
    def __init__(self, w):
        self.w = w
        self.n = len(w)
        for i in range(1,self.n):           ### creating cumulative sum list 
            w[i] += w[i-1]          
        self.lastElementCumulative = self.w[-1]  ###we get the last element of cumulative
    def pickIndex(self):
        target = random.randint(1,self.lastElementCumulative) ###now randomly generate a number between these two inclusive 
        #for target in range(1,26,1):  ### for testing if everything works as expected
    
        left=0
        right =self.n-1
        while left<right-1:
            mid = (left+right)/2
            if self.w[mid]==target:
                # print("mid",mid)
                return mid
                break
            elif self.w[mid]>target:
                right=mid
            else:
                left=mid  
        # print("=======")
        # print("target",target)
        # print("w",self.w)
        if self.w[left]>=target:    #### equality part gives rise to errors, target can be smaller than first entry too
            print("left",left)
            return left 
        else:
            print("right",right)
            return right
======================================================================
346. Moving Average from Data Stream

class MovingAverage(object):
    def __init__(self, size):
        self.list1 = []
        self.size=size
    
    def next(self, val):
        if len(self.list1)<self.size:
            self.list1.append(val)
        else:
            self.list1.pop(0)
            self.list1.append(val)
        return float(sum(self.list1))/len(self.list1)


======================================================================
1570. Dot Product of Two Sparse Vectors
Learn how objects are instantiated and METHODS are created. 
created a dictionary object which stores index:value of non zero objects


class SparseVector:
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.dict1={}
        
        for i in range(len(nums)):
            if nums[i]!=0:
                self.dict1[i]=nums[i]       #### dict1 is a class variable here because it was defined in init
            
    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec):
        """
        :type vec: 'SparseVector'
        :rtype: int
        """
        
        dotProd=0
        
        for x in self.dict1:
            if x in vec.dict1:          ### vec is a sparse object and the dict related to it needs to be called like this 
                dotProd+=self.dict1[x]* vec.dict1[x]
        return dotProd
---------------------------------------------------------------------

class SparseVector:
    def __init__(self, nums):
        self.arr1=[]
        
        for i in range(len(nums)):
            if nums[i]!=0:
                self.arr1.append((i,nums[i]))
            
    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec):
        dotProd=0
        
        i=0
        j=0
        while i<=len(self.arr1)-1 and j<=len(vec.arr1)-1:
            if self.arr1[i][0]==vec.arr1[j][0]:
                dotProd+=self.arr1[i][1]*vec.arr1[j][1]
                i+=1            ### dont forget increasing in this case
                j+=1
            elif self.arr1[i][0]<vec.arr1[j][0]:
                i+=1
            elif self.arr1[i][0]>vec.arr1[j][0]:
                j+=1
                
        return dotProd
        
Using tuples if someone says dict method is bad due to hashing problems?? what?
Update from recent FB onsite, interviewer didn't accept the HASHMAP solution and wanted to see the 2 pointers solution,
in addition he also came up with a follow up question, what would you do if one of the vectors werent fully "sparse" -> hint use binary search

Follow up: What if only one of the vectors is sparse?

https://leetcode.com/problems/dot-product-of-two-sparse-vectors/discuss/1522271/Java-O(n)-solution-using-Two-Pointers-with-detailed-explanation-and-Follow-up
If one of the vector is sparse, then we can use the third approach, pick upon the indices of the sparse vector and then binary search the indices in the second vector.
I don't understand binary search on the sparse vector. why not iterate on sparse vec and simply access index of the non sparse.
i guess the non sparse is still stored in tuple format. now iterate on sparse one because less and binary search on non sparse

If the length of one sparse vector's non-zero element is much greater than the other one's, we could use binary search on the long sparse vector.
======================================================================
287. Find the Duplicate Number
### add to dict if dup found return
        dict1={}
        
        for i in range(len(nums)):
            if nums[i] not in dict1:
                dict1[nums[i]]=1
            else:
                return nums[i]
### saving dict space by changing sign 
for i in range(len(nums)):
    if nums[abs(nums[i])]>0:   ## will this go out of index??
        nums[abs(nums[i])]= -1*nums[abs(nums[i])]
    else:
        return abs(nums[i])
======================================================================
43. Multiply Strings
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
Tag: Math, String 
---------------------------------------------------------------------

DIFFICULT TRICK TO REMEMBER : 
DO the type of multiplication we do on paper.num1[i] * num2[j] will be placed at indices [i + j, i + j + 1]




https://leetcode.com/problems/multiply-strings/discuss/17605/Easiest-JAVA-Solution-with-Graph-Explanation


class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        m=len(num1)
        n=len(num2)
        
        ans=[0]*(m+n)   						 ### assigning for one extra digit 
        for i in range(len(num1)-1,-1,-1):       ### coming backwards 
            for j in range(len(num2)-1,-1,-1):

                mul=int(num1[i])*int(num2[j]) 
                sum1=ans[i+j+1]+mul              ### YOU ADD THE MULTIPLICATION TO I+J+1 and NOT I+J   
                
                ans[i+j]+=sum1/10                 ### I+J gets the carry added to the previous number 
                ans[i+j+1]=sum1%10                ### I+J+1 just gets overall
        
        for i in range(len(ans)):              ### removing leading edges using this 
            if ans[i]==0:
                pass
            else:
                break
        

        return ''.join([str(x) for x in ans[i:]])

======================================================================
593. Valid Square
Done
Logic: I dont know why so stupid solutions are given in leetcode. I want to calculate 6 distances and sort. bottom 4 should be same. top 2 should be same. top should be diagonal of bottom
Remove edge case points shouldnt be same.

class Solution(object):
    def distance(self,p1,p2):
        return (p2[0]-p1[0])**2+(p2[1]-p1[1])**2
    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        """
        
        d1=self.distance(p1,p2)
        d2=self.distance(p1,p3)
        d3=self.distance(p1,p4)
        d4=self.distance(p2,p3)
        d5=self.distance(p2,p4)
        d6=self.distance(p3,p4)
        
        list1=sorted([d1,d2,d3,d4,d5,d6])
        if sum(list1)==0:   ### remove edge case of all points being the same
            return False
        
        for i in range(4):
            if list1[i]!=list1[0]:
                return False
        if list1[-1]!=list1[-2]:
            return False
        if list1[-1]!=2*list1[0]:
            return False
        return True
====================================================================== 
2013. Detect Squares
Now the square here IS AXIS ALIGNED. 
### the trick is that you only need to consider diagonal points where dx and dy is same 
### squares are axis aligned, we use that info to check top and right cornerns
### duplicates points are allowed so for a given point we can choose any of its duplicates. MULTIPLICATION
### Remove Zero area by either comparing diagonal points directly or by checking if the abs difference 0
class DetectSquares(object):
    def __init__(self):
        self.dict1={}
    
    def add(self, point):
        """
        :type point: List[int]
        :rtype: None
        """
        if tuple(point) not in self.dict1:
            self.dict1[tuple(point)]=1
        else:
            self.dict1[tuple(point)]+=1
        
    def count(self, point):
        """
        :type point: List[int]
        :rtype: int
        """
        
        count=0
        for x,y in self.dict1.keys():    ###iterating and searching for diagonal points
            if abs(point[0]-x) == abs(point[1]-y) and (x,y)!=tuple(point) and (point[0],y) in self.dict1 and (x,point[1]) in self.dict1:
                count+=self.dict1[(point[0],y)]*self.dict1[(x,point[1])]*self.dict1[(x,y)]
        return count
====================================================================== 
836. Rectangle Overlap
For rectangle overlap, we need an overlap in both x and y.
So I simply create the condn for non overlap as it is easier.
Did it myself lol
class Solution(object):
    def isRectangleOverlap(self, rec1, rec2):
        x11=rec1[0]
        x12=rec1[2]
        y11=rec1[1]
        y12=rec1[3]
        
        x21=rec2[0]
        x22=rec2[2]
        y21=rec2[1]
        y22=rec2[3]
        if x22<=x11 or x21>=x12:
            xOverlap=False
        else:
            xOverlap=True
        
        if y22<=y11 or y21>=y12:
            yOverlap=False
        else:
            yOverlap=True
        return xOverlap and yOverlap
---------------------------------------------------------------------


====================================================================== 
268. Missing Number
Done
simply subtract from total sum- n*n+1/2
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return len(nums)*(len(nums)+1)/2-sum(nums)     
======================================================================           
            
            
        
        



++++++++++++++++++++++++++++++++++++++
+++++++Group : HEAP +++
++++++++++++++++++++++++++++++++++++++

1046. Last Stone Weight
======================================================================
class Solution(object):
    def lastStoneWeight(self, stones):
        stones=[-1 * weight for weight in stones]
        heapq.heapify(stones) ## now stones is a max heap ## O(N) operation
        
        while len(stones)>1:        ### WE CAN DO THIS N/2 times if all equal
            y=heapq.heappop(stones) ##negative   ## O(logn) operation as we have to rebalance ##heap[0] is O(1)
            x=heapq.heappop(stones) ##negative
            if y==x:
                pass
            else:
                heapq.heappush(stones,(y-x))    ## O(logn) operation
        
        if len(stones)==1:
            return stones[0]*-1     ## dont forget signs
        else:
            return 0
Time complexity : O(NlogN) 
Space complexity : O(1) 
======================================================================
703. Kth Largest Element in a Stream
class KthLargest(object):   ##TLE

    def __init__(self, k, nums):
        self.k=k
        nums=[-1*x for x in nums]
        heapq.heapify(nums) #O(N)  ## max heap
        self.nums=nums
    
    def add(self, val):
        logging=False
        heapq.heappush(self.nums,-1*val)
        list1=[]
        for i in range(self.k-1):
            list1.append(heapq.heappop(self.nums))
        if logging: print(self.nums)
        a= self.nums[0]*-1
        for x in list1:
            heapq.heappush(self.nums,x)
        if logging: print(self.nums)
        return a        
---------------------------------------------------------------------    
If you use a min heap and and keep the heap at size k. 
That way, the smallest element in the heap (the one we can access in O(1) will always be the kth largest element
class KthLargest(object):

    def __init__(self, k, nums):
        self.k=k
        
        self.heap1=[]  ### WE CREATE A MIN HEAP AS WE WANT KTH MAX ELEMENT 
        for i in nums:
            heapq.heappush(self.heap1,i)    
            if len(self.heap1)==k+1:
                heapq.heappop(self.heap1) 
        ### After this I have a heap of exactly size k in ideal case. 
    
    def add(self, val):
        
        if len(self.heap1)==self.k-1:  #### only this is possible, otherwise 
                                       #### we wont have the kth maximum as guaranteed
            heapq.heappush(self.heap1,val) 
        elif len(self.heap1)==self.k:
            heapq.heappush(self.heap1,val)  ##O(logn)       ### WE PUSH FIRST, POP LATER ## IMPORTANT
            ## heap size is k+1
            heapq.heappop(self.heap1) 
            ## heap size k again 
        
        if self.heap1:
            return self.heap1[0]  ## min heap of size k, 0th element gives k largest number!! Remember
        else:
            return 0
======================================================================




Task Scheduler
Tag: Heap 
======================================================================
Other logics

1st logic underflow -- total task/ (no of distimct task)  --- This is wrong because it is possible for a task to end.

2nd logic 
underflow 
number of slots = freq of most common element ??
overflow
can we calculate this ?



logic:
1. put tasks into heap
2. draw most common n+1 from heap and reduce heap values accordingly.
3. if you draw most common, the conditions of distance will be met (both in underflow and overflow)
4. The last cycle can be incomplete so we remove the blanks from the last cycle.


from collections import Counter
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        tasks_to_counts = Counter(tasks)
        index = 0
        while tasks_to_counts:
            task_list = tasks_to_counts.most_common(n+1)    ### drawing n+1 most frequent each time
            for task, count in task_list:
                tasks_to_counts[task] -= 1
                if not tasks_to_counts[task]:
                    del tasks_to_counts[task]
            index+=n+1
        
        return index-(n+1-len(task_list))     ### if the last lo
========================================================================================================================
846. Hand of Straights
Tag: Greedy, Heap
1. We need to create groups of k CONSECUTIVE numbers.
2. The min of the list has to start the list because it wont fit anywhere else.
3. Find min and start the consective sequence up to k numbers. 
4. Once this sequence is done, find the next min 
5. We keep finding mins, again and again, we can optimize by using a min heap 
6. Once we are done with all numbers and all they all fit into groups, return True else return False 
========================================================================================================================
I have to use two while loops one for total count , one for adding elements to group and checking
1. Cases where i return False: Gap in the elements, first+1 wont be found in hashmap, less than the next multiple of groupsize(will be always less lol), again missing
2. if the loop runs properly without missing, we can return True

Correct but TLE

class Solution(object):
    def isNStraightHand(self, hand, groupSize):
        if len(hand)%groupSize!=0:
            return False
        dict1=collections.Counter(hand)
        
        totalCount=0
        while totalCount<len(hand):
            heap=dict1.keys()
            #print("heap",heap)
            heapq.heapify(heap) ## min heap
            first=heapq.heappop(heap) ## get minimum element
            #print("first",first)
            #print("dict1",dict1)
            if dict1[first]==1:
                del dict1[first]
            else:
                dict1[first]-=1
            count=1
            while count<groupSize:
                if first+1 in dict1:
                    if dict1[first+1]==1:
                        del dict1[first+1]
                    else:
                        dict1[first+1]-=1
                    count+=1
                    first=first+1
                else:
                    #print("first+1",first+1)
                    #print(dict1,"dict1")
                    return False
            totalCount+=count
            #print("totalCount",totalCount)
        return True  
========================================================================================================================
692. Top K Frequent Words
Conceptually simple, 
## asking for k most frequent strings 
## k largest,maintain a min heap of size k 
Now strings are sorted in lexico order already. 
For the heap sorting we will need to implement a special class and override the heap sorting internals. 
This is too much. 


class Count:
    def __init__(self, count, keyword):
        self.count = count
        self.keyword = keyword
    # override the comparison operator - heapq internally uses this for comparison
    def __lt__(self, other):
        if self.count==other.count:
            return self.keyword>other.keyword  ## keyword if count is equal 
        return self.count<other.count ## count by default ## this returns a BOOL VALUE 

class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        dict1=collections.Counter(words) ##O(n)
        
        heap1=[]            ## min heap
        for key in dict1.keys():
            heapq.heappush(heap1,Count(dict1[key],key)) ## nlogk 
            if len(heap1)==k+1:
                heapq.heappop(heap1)
        
        heap1.sort(key=lambda x:(-1*x.count,x.keyword))
        #print(heap1)
        return [x.keyword for x in heap1]      





++++++++++++++++++++++++++++++++++++++
+++++++Group : Graph +++
++++++++++++++++++++++++++++++++++++++

https://leetcode.com/discuss/general-discussion/655708/Graph-For-Beginners-Problems-or-Pattern-or-Sample-Solutions

DIFFERENT TYPE OF GRAPHS
===============================
UNDIRECTED
DIRECTED 
WEIGHTED 




GRAPHS AND THE ACROBATICS WE KNOW
======================================================================
ACROBATIC 1. We can count the number of nodes in graph by doing DFS/BFS
ACROBATIC 2. We can count how many levels in the graph by doing BFS. Using this we can max height, min height of a TREE also. 
ACROBATIC 3. We can count the number of unconnected components by starting DFS at any point, marking visited. We try to run DFS on all nodes, but some might be visited from before.  The number of times we are successfully able to start dfs is the count of unconnected components.

ACROBATIC 4. We know if there is a loop/cycle in a directed graph. -- TOPOLOGICAL SORT
ACROBATIC 5: We can tell if starting from a point we can reach leaf node - Single point dfs_topo
ACROBATIC 5. We know if there is a loop/cycle in a directed graph. -- UNION FIND
ACROBATIC 6. We can find the maximum path sum in all directions, given a starting point or we can maximum path sum in all directions from all points.
ACROBATIC 7. We can go from root to leaves in a graph/tree using DFS. can store the path or sum or whatever






DFS tree 

DFS tree can of three types: Preorder, Inorder and Postorder

Preorder - root, left right 2. Inorder left root right 3. post order left right root
Observe the location of root each time 


ALGO_NAME: TREE_STYLE_DFS_TAIL_RECURSIVE--I HARDLY USE THIS, I USE DFS ITERATIVE IT ITS PLACE
----------------------------------
self.ans=[]
def dfsR(self,root):
    if not root:
        return 
    neighbors=[root.left,root.right]
    for neighbor in neighbors:  ### FIFO 
        self.dfsR(neighbor)
    self.ans.append(root.val)



DFS tree (Using stack)          ### stack and recursion are essentially same thing
-------------------------------
Preorder Traversal -- Natural DFS . 
Postorder Traversal-- small change to Preorder
In order Traversal (Modified DFS) Memorize different than preorder and post order DFS




ALGO_NAME: TREE_STYLE_BFS_ITERATIVE
-------------------------
class Solution(object):
    def levelOrder(self, root):
        queue=[root]
        ret=[]
        while queue:
            current=queue.pop()
            for i in [current.left,current.right]:
                if i:
                    queue.insert(0,i)
            ret.append(current.val)
        return ret
-------------------------


ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK
-------------------------------------
Run in jupyter
vertexList = [0, 1, 2, 3, 4, 5, 6]
edgeList = [(0,1), (0,2), (1,0) , (1,3) , (2,0) , (2,4) , (2,5) , (3,1), (4,2) , (4,6), (5,2), (6,4)]
### Iterative DFS
class Solution(object):
    def dfs(self,vertexList,edgeList ,start):
        visited = {}

        adjacencyList = {vertex:[] for vertex in vertexList}
        for edge in edgeList:                       
            adjacencyList[edge[0]].append(edge[1])   ### ensure all edges are present and you dont need to add vice versa

        pathList=[]
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacencyList[current]:  ### LIFO 
                if neighbor not in visited:    ### if already visited then we do nothing
                    stack.append(neighbor)
            visited[current]=1
            pathList.append(current)
        
        return pathList
a=Solution()
print(a.dfs(vertexList,edgeList, 0))
========================================================================================================================
I usually do the iterative dfs, but recursive DFS is crucial for some questions and for topological sorting 

ALGO_NAME: GRAPH_STYLE_DFS_TAIL_RECURSION
-------------------------------------
two decisions of recursion: extra function  -- yes because we dont want adjacencyList and visited each time.
											   Anything else?
						    which ones global -- visited needs to be global (it needs to be accessed through all stacks)

### Recursive DFS- Run in jupyter
vertexList = [0, 1, 2, 3, 4, 5, 6]
edgeList = [(0,1), (0,2), (1,0) , (1,3) , (2,0) , (2,4) , (2,5) , (3,1), (4,2) , (4,6), (5,2), (6,4)]

class Solution(object):
    def dfs(self,vertexList,edgeList ,start):
        self.visited = {}

        adjacencyList = {vertex:[] for vertex in vertexList}
        for edge in edgeList:                       
            adjacencyList[edge[0]].append(edge[1])   ### ensure all edges are present and you dont need to add vice versa
        self.pathList=[]
        self.dfsR(start,adjacencyList)
        return self.pathList

    def dfsR(self,current,adjacencyList):
        if current not in self.visited:
            self.visited[current]=1
            self.pathList.append(current)
            for neighbor in adjacencyList[current]:  ### FIFO 
                self.dfsR(neighbor,adjacencyList)
========================================================================================================================




========================================================================================================================



ALGO_NAME: GRAPH_STYLE_BFS_ITERATIVE
------------------------- 
https://www.youtube.com/watch?v=QkEOGoUar3g

vertexList = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
edgeList = [(0,1), (1,2), (1,3), (3,4), (4,5), (1,6)]
graphs = (vertexList, edgeList)

def bfs(graph, start):
    vertexList, edgeList = graph
    visitedList = []
    queue = [start]
    adjacencyList = [[] for vertex in vertexList]

    # fill adjacencyList from graph
    for edge in edgeList:
        adjacencyList[edge[0]].append(edge[1])

    # bfs
    while queue:
        current = queue.pop()
        for neighbor in adjacencyList[current]:
            if not neighbor in visitedList:
                queue.insert(0,neighbor)
        visitedList.append(current) ### VISTED MARKING HAPPENS AFTER POPPING
    return visitedList

print(bfs(graphs, 0))
--------------------------------------------------- 
-------------------------------------------------- 

++++++++++++++++++++++++++++++++++++++
+++++++Group : DFS +++
++++++++++++++++++++++++++++++++++++++
145. Binary Tree Postorder Traversal
---------------------------------------------------------------------
Done all 3 
---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_TAIL_RECURSION

Simple Tail recursion carrying list 
class Solution(object):
    def postorderTraversalR(self, root,list1):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return
        
        self.postorderTraversalR(root.left,list1)
        self.postorderTraversalR(root.right,list1)
        list1.append(root.val)
        
        return list1

    
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        
        return self.postorderTraversalR(root,[])
---------------------------------------------------------------------------
ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION


class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        
        x=self.postorderTraversal(root.left)
        y=self.postorderTraversal(root.right)
        
        return x+y+[root.val]
---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK 

class Solution(object):
    logging=False
    def dfs(self,root):
        stack=[root]
        ret=[]
        while stack:
            current=stack.pop()
            neighbors=[current.left,current.right]
            #### LOOOK ATT THEE ORDERR !!!! VERY IMPORTANT AS IT WILL GET INVERTED !!!!! WHY ?              BECAUSE WE ARE POPPING LATER  ## THIS ORDER IS DIFFERENT FROM PREORDER
            for neighbor in neighbors:
                ## neighbor breakdown not needed
                if neighbor:
                    stack.append(neighbor) 
            ret.insert(0,current.val) ### LINE OF CHANGE FROM PREORDER!!!
            
        return ret
    def postorderTraversal(self, root):
        if not root:  #edge case not base case
            return [] 
        return self.dfs(root)
=========================================

144. Binary Tree Preorder Traversal
---------------------------------------------------------------------
Done all 3 
---------------------------------------------------------------------
ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION

class Solution(object):
    def preorderTraversal(self, root):
        if not root:
            return []
        
        x=self.preorderTraversal(root.left)
        y=self.preorderTraversal(root.right)
        
        return [root.val]+x+y

-------------------------------------------------- 
ALGO_NAME: TREE_STYLE_TAIL_RECURSION (CARRIED LIST) 

class Solution(object):
    def preorderTraversalR(self, root,list1):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return
        list1.append(root.val)
        self.preorderTraversalR(root.left,list1)
        self.preorderTraversalR(root.right,list1)
        
        return list1

    
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        
        return self.preorderTraversalR(root,[])
------------------------- 
ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK
preorder is natural DFS

class Solution(object):
    logging=False
    def dfs(self,root):
        stack=[(root)]
        ret=[]
        while stack:
            current=stack.pop()
            neighbors=[current.right,current.left]
            #### LOOOK ATT THEE ORDERR !!!! VERY IMPORTANT AS IT WILL GET INVERTED !!!!! WHY ? BECAUSE WE ARE POPPING LATER 
            for neighbor in neighbors:     
                if neighbor:
                    stack.append(neighbor) 
            ret.append(current.val)
        return ret

    def preorderTraversal(self, root):
        if not root:  #edge case not base case
            return [] 
        return self.dfs(root)
==================================================================================


==================================================================================

733. Flood Fill
---------------------------------------------------------------------
Done
Always tricks me this question because of the following.

I ALWAYS FORGET STORING THE ORIGINAL COLOR AND THIS QUESTION GIVES ME A LOT OF PAIN BECAUSE OF THAT!!
NO SR AND SC ARE NOT ENOUGH!! YOU NEED TO STORE THE ORIGINAL!!
======================================================================
ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK

class Solution(object):
    logging=False
    def dfs(self,image, sr, sc, newColor):
        m=len(image)
        n=len(image[0])
        visited={}
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        stack=[(sr,sc)] ### Remeber to add cordinates not values LOL
        originalColor=image[sr][sc]
        while stack:
            x,y=stack.pop()
            neighbors=[(x+addX,y+addY) for addX,addY in dirs]
            for neighbor in neighbors:
                nextX,nextY=neighbor
                if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and neighbor not in visited and neighbor not in stack and image[nextX][nextY]==originalColor:
                    stack.append(neighbor)
                
            image[x][y]=newColor
            visited[(x,y)]=1
        return image
        
    def floodFill(self, image, sr, sc, newColor):
        return self.dfs(image, sr, sc, newColor)
---------------------------------------------------------------------
Space Optimized: 
we can remove visited by using coloring as VISITED MARK but need to add edge case here!!!!

class Solution(object):
    logging=True
    def dfs(self,image, sr, sc, newColor):
        m=len(image)
        n=len(image[0])
        #visited={}
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        stack=[(sr,sc)] ### Remeber to add cordinates not values LOL
        originalColor=image[sr][sc]
        if newColor==originalColor: ###edge case has to be resolved after seeing print statements
            return image
        
        while stack:
            x,y=stack.pop()
            neighbors=[(x+addX,y+addY) for addX,addY in dirs]
            if self.logging: print("x,y",x,y)
            if self.logging: print("neighbors",neighbors)
            for neighbor in neighbors:
                nextX,nextY=neighbor
                if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and neighbor not in stack and image[nextX][nextY]==originalColor:
                    stack.append(neighbor)
            #visited[(x,y)]=1  
            image[x][y]=newColor 
            ## color change happens after stack pop ## this acts as visited condition as 
            ## popped items are newColored and cant be added back to stack 
            if self.logging: print("stack",stack)
            #if self.logging: print("visited",visited)
        return image
        
    def floodFill(self, image, sr, sc, newColor):
        return self.dfs(image, sr, sc, newColor)
        
---------------------------------------------------------------------
## This difference is that we mark VISITED BEFORE ADDITION TO STACK OR AFTER COMING OUT!!!! We wont use this code.
##Just for show 

class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        stack=[[sr,sc]]
        prev=image[sr][sc] ###storing the original color 
        image[sr][sc]=newColor    #### This is important as we changing color before stack addition the current one is colored here 
        
        if prev==newColor:
            return image
        
        dirs=[[1,0],[-1,0],[0,1],[0,-1]]
        while stack:
            current=stack.pop()
            neighbors=[[current[0]+dir[0],current[1]+dir[1]] for dir in dirs]
            for x in neighbors:
                nextX=x[0]
                nextY=x[1]
                if nextX>=0 and nextX<=len(image)-1 and nextY>=0 and nextY<=len(image[0])-1 and image[nextX][nextY]==prev:  ### again comparing tp prev and not to sr,sc curent value bcoz that might be changed
                    image[nextX][nextY]=newColor   ### this is acting as the visited condition
                    stack.append([nextX,nextY])
        
        return image
======================================================================
323.No of connected components in an Undirected Graph
Tags: Graph, DFS
---------------------------------------------------------------------
Done dfs stack
1. Essentially do a dfs on each node and maintain a count. why ?
Each time we finish exploring a connected component, we can find another vertex that has not been visited yet, and start a new DFS from there. The number of times we start a new DFS will be the number of connected components.
---------------------------------------------------------------------
Method1: ALGO_NAME: GRAPH_STYLE_DFS_TAIL_RECURSION -- I dont use this

Recursion decisions: 1. two functions because i dont want the logic on top to be initialized each time
                     2. Two self variables (visited and dict). Why ? because both need to be accesible 
                     visited is modified through the recursion stacks and we want to keep all changes. (so global)
                     dict1 is just used by the new function. it doesnt change through the recursion stacks so both global and passing it in the function will work.   


class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        self.dict1={}
        for i in range(n):
            self.dict1[i]=[]
        
        for i in range(len(edges)):
            if edges[i][0] in self.dict1:
                self.dict1[edges[i][0]].append(edges[i][1])
                self.dict1[edges[i][1]].append(edges[i][0])
	
        count1=0
        self.visited={}
        for i in range(n):
            if i not in self.visited:
                count1+=1
                self.dfs(i)
                
        return count1

    ## add nodes that are connected to start node
    def dfs(self,i):
        self.visited[i]=1
        for j in self.dict1[i]:
            if j not in self.visited:
                self.dfs(j)
        

Method2: ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK
Algo cant be followed super closely but I keep a list of what needs to be added in GRAPH from TREE
5 items

# dirs, dimension, boundary condn, visited, stack
class Solution(object):
    logging=False
    def dfs(self,node,adjacencyList):
        stack=[node]
        while stack:
            current=stack.pop()
            neighbors=adjacencyList[current]
            if self.logging: print("current",current)
            if self.logging: print("neighbors",neighbors)
            for neighbor in neighbors:
                if neighbor not in stack and neighbor not in self.visited:
                    stack.append(neighbor)
            self.visited[current]=1
            if self.logging: print("stack",stack)
            if self.logging: print("self.visited",self.visited)
                
    def countComponents(self, n, edges):
        adjacencyList={i:[] for i in range(n)}
        for edge in edges:
            adjacencyList[edge[0]].append(edge[1])
            adjacencyList[edge[1]].append(edge[0])  ### I had to reverse order TOO WHY ?
                                                    ### ALWAYS ADD THIS WHEN EDGES ARE GIVEN
        print("adjacencyList",adjacencyList)
        count=0
        self.visited={}
        for node in adjacencyList.keys():
            if node not in self.visited:
                count+=1
                self.dfs(node,adjacencyList)
        return count
======================================================================
547. Friend Circles/Number of Provinces
---------------------------------------------------------------------
ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK
1. Think of this excatly as number of connected components
2. We have N nodes and the adjacency list has been given. Now start a dfs on each node and mark connected nodes vsited.
Each time we finish exploring a connected component, we can find another vertex that has not been visited yet, and start a new DFS from there. The number of times we start a new DFS will be the number of connected components.
3. Never forget that while writing the dfs you have to consider the Currents neighbors. Making mistake here. 
4. Dont think of this like no of Islands!!! 

This is not like GRID PROBLEM WHERE YOU CAN MOVE IN DIRS. 
THIS IS NOT A GRID PROBLEM WHERE YOU CAN ADD x,y of grid in visited. WE ONLY NEED TO ADD X IN GRID HERE.
AND do dfs on X instead of x,y
---------------------------------------------------------------------
# visited, stack
class Solution(object):
    logging=False
    def dfs(self,i,isConnected):
        m=len(isConnected)
        n=len(isConnected[0])
        stack=[i]
        while stack:
            x=stack.pop()
            neighbors=[j for j in range(n) if isConnected[x][j]==1 and j!=x]
            if self.logging: print("x",x)
            if self.logging: print("neighbors",neighbors)
            for neighbor in neighbors:
                nextX=neighbor
                if neighbor not in self.visited and neighbor not in stack:
                    stack.append(neighbor)
            self.visited[x]=1
            #self.visited[(y,x)]=1
            if self.logging: print("self.visited",self.visited)
            if self.logging: print("stack",stack)
            
    def findCircleNum(self, isConnected):
        m=len(isConnected)
        n=len(isConnected[0])
        self.visited={}
        count=0
        for i in range(m):
            if i not in self.visited:
                if self.logging: print("entering dfs",(i),"count",count)
                count+=1
                self.dfs(i,isConnected)
        return count
======================================================================
417. Pacific Atlantic Water Flow
Tags: DFS
Question very similar to surrounded regions
---------
APPROACH:
## DFS from each cell and see if it reaches both oceans
## This is brute force 
## optimization 
## what about marking as P and A and once they reach here BFS is done. 
## MORE INNOVATIVE
## How about REVERSE DFS FROM EDGES !! WOW THIS WILL WORK 


1. DFS from edges 
2. Dont try to manipulate the matrix, create a separte visited dict 
3. between pacific and atlantic you need two different dicts but the visited dict should be same for between pacific calls 
4. Simply take intersection of Pacific and atlantic after this for answer.

------------------------------------------------------
ALGO_NAME: GRAPH_STYLE_DFS_TAIL_RECURSION

class Solution(object):
    def dfs(self,x,y,visitedDict,heights):
        m=len(heights)
        n=len(heights[0])
        stack=[(x,y)] ###dont make this list as lists cant be added to dict
        while stack:
            current=stack.pop()
            x=current[0]
            y=current[1]
            neighbors=[]
            if x+1<=m-1:
                neighbors.append((x+1,y))
            if x-1>=0:
                neighbors.append((x-1,y))
            if y+1<=n-1:
                neighbors.append((x,y+1))
            if y-1>=0:
                neighbors.append((x,y-1))
            for neighbor in neighbors:
                if heights[neighbor[0]][neighbor[1]]>=heights[x][y] and neighbor not in visitedDict:
                    stack.append(neighbor)
            visitedDict[current]=1 
        return visitedDict
    
    def pacificAtlantic(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: List[List[int]]
        """
        m=len(heights)
        n=len(heights[0])
        
        visitedPacific={}
        for j in range(n):
            visitedPacific=self.dfs(0,j,visitedPacific,heights)
        for i in range(m):
            visitedPacific=self.dfs(i,0,visitedPacific,heights)
        
        visitedAtlantic={}
        for j in range(n):
            visitedAtlantic=self.dfs(m-1,j,visitedAtlantic,heights)
        for i in range(m):
            visitedAtlantic=self.dfs(i,n-1,visitedAtlantic,heights)
        
        return list(set(visitedPacific.keys()).intersection(visitedAtlantic.keys()))
------------------------------------------------------ 
THis is the proper solution-- ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK
class Solution(object):
    logging=False
    def dfs(self,i,j,heights,visited):
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        m=len(heights)
        n=len(heights[0])
        stack=[(i,j)]
        while stack:
            x,y=stack.pop()
            neighbors=[(x+addX,y+addY) for addX,addY in dirs]
            # if self.logging: print("current",x,y)
            # if self.logging: print("neighbors",neighbors) 
            for neighbor in neighbors:
                nextX,nextY=neighbor
                if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and neighbor not in visited and neighbor not in stack and heights[nextX][nextY]>=heights[x][y]:
                    stack.append(neighbor)
            visited[(x,y)]=1
            # if self.logging: print("visited",visited)
            # if self.logging: print("stack",stack)

    def pacificAtlantic(self, heights):
        m=len(heights)
        n=len(heights[0])
        self.visitedP={}
        self.visitedA={}
        # self.visitedP=self.dfs(0,0,heights,self.visitedP)
        # print(self.visitedP)
        for j in range(n):
            self.dfs(0,j,heights,self.visitedP)
            self.dfs(m-1,j,heights,self.visitedA)
        for i in range(m):
            self.dfs(i,0,heights,self.visitedP)
            self.dfs(i,n-1,heights,self.visitedA)
        
        # print(self.visitedP)
        # print(self.visitedA)
        ans=[]
        for i in self.visitedP:
            if i in self.visitedA:
                ans.append(i)
        return ans   
                   
======================================================================
130. Surrounded Regions
Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.
A region is captured by flipping all 'O's into 'X's in that surrounded region.
------------------------------------------------------------------------------------------------
Done
start dfs from each edge and mark as special char. only these wont be captured. rest will be captured
Note we can only start from an O
------------------------------------------------------------------------------------------------
#####AFTER POP VISITED MARKING
ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK
class Solution(object):
    logging=True
    def dfs(self,i,j,board):
        if board[i][j]!='O':        ### dont miss this You can only start from an O
            return 
        m=len(board)
        n=len(board[0])
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        stack=[(i,j)]
        #visited={}
        while stack:
            x,y=stack.pop()
            neighbors=[(x+addX,y+addY) for addX,addY in dirs]
            for neighbor in neighbors:
                nextX,nextY=neighbor
                if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and board[nextX][nextY]=="O" and neighbor not in stack:
                    stack.append(neighbor)
            board[x][y]="*"
                    
    def solve(self, board):
        m=len(board)
        n=len(board[0])
    
        for i in range(m):
            self.dfs(i,0,board)
            self.dfs(i,n-1,board)
        for j in range(n):
            self.dfs(0,j,board)
            self.dfs(m-1,j,board)
        
        #print(board)
        
        for i in range(m):
            for j in range(n):
                if board[i][j]!="*":
                    board[i][j]='X'
                else:
                    board[i][j]='O'     
------------------------------------------------------------------------------------------------
#BEFORE POP VISITED MARKING -- I dont use this 
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if len(board)==0:
            return 
        
        self.dirs=[[1,0],[-1,0],[0,1],[0,-1]]
        for j in range(len(board[0])):
            if board[0][j]=='O':
                self.dfs(board,0,j)
            if board[len(board)-1][j]=='O':
                self.dfs(board,len(board)-1,j)
            
        for i in range(len(board)):
            if board[i][0]=='O':
                self.dfs(board,i,0)
            if board[i][len(board[0])-1]=='O':
                self.dfs(board,i,len(board[0])-1)
        #return board       
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j]=='*':
                    board[i][j]='O'
                else:
                    board[i][j]='X'
        return board
                    
    def dfs(self,board,i,j):
        
        stack=[[i,j]]
        board[i][j]='*' ################### DONTTTT FORGETTTTTTT THIS AS YOU ARE MARKING VISIT BEFORE QUEUEEEUEUEUEUE!!
        while stack:
            current=stack.pop()
            neighbors=[[current[0]+dir[0],current[1]+dir[1]] for dir in self.dirs]
            for x in neighbors:
                nextX=x[0]
                nextY=x[1]
                if nextX>=0 and nextX<=len(board)-1 and nextY>=0 and nextY<=len(board[0])-1 and board[nextX][nextY]=='O':
                    stack.append([nextX,nextY])
                    board[nextX][nextY]='*'
======================================================================  

++++++++++++++++++++++++++++++++++++++
+++++++Group : Graph,BFS +++
++++++++++++++++++++++++++++++++++++++







994. Rotting Oranges

logic 
---------------------------------------------------------------------
1. essentially its a BFS but multi point BFS because we are starting from several pints so add them in the stack in the beginning only. Standard BFS. ADD VISITED AFTER POP.
2. Simply count the levels of BFS. DONT FUCK IT UP BY ALLOWING DUPS.
3. Fresh counting is not "REALLY" needed. Why ? BFS automatically stops, no need for fresh to be zero or something.
4. We can loop once again to check for remaining fresh
5. 1 edge case which needs to dealth with -- 0 starting fresh
---------------------------------------------------------------------
ALGO_NAME: GRAPH_STYLE_BFS_ITERATIVE
class Solution(object):
    logging=False
    def orangesRotting(self, grid):
        m=len(grid)
        n=len(grid[0])
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        queue=[]
        for i in range(m):
            for j in range(n):
                if grid[i][j]==2:
                    queue.append((i,j))
        level=0
        while queue:
            for i in range(len(queue)):
                x,y=queue.pop()
                neighbors=[(x+addX,y+addY) for addX,addY in dirs]
                if self.logging: print("current",x,y)
                if self.logging: print("neighbors",neighbors)
                for neighbor in neighbors:
                    nextX,nextY=neighbor
                    if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and grid[nextX][nextY]==1 and  neighbor not in queue:
                        queue.insert(0,neighbor)
                        
                if grid[x][y]!=2:       ### I made a mistake here by not checking
                    grid[x][y]=2
                if self.logging: print("queue",queue)
                        
            level+=1        ##0,1,2
        
        freshCount=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    freshCount+=1
        if freshCount>0:
            return -1
        if level==0:        ## this happens when there are no fresh to begin with and the entire loop skips
            return 0
        else:
            return level-1           
---------------------------------------------------------------------
#WE can avoid the second iteration for checking fresh if we keep a fresh count for the first time 
## BE CAREFUL WHEN TO SUBTRACT FRESH, MAKING A MISTAKE HERE by subtracting always

class Solution(object):
    logging=False
    def orangesRotting(self, grid):
        m=len(grid)
        n=len(grid[0])
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        queue=[]
        freshCount=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==2:
                    queue.append((i,j))
                if grid[i][j]==1:
                    freshCount+=1
        
        if freshCount==0:   ##edge case
            return 0
                 
        level=0
        while queue:
            for i in range(len(queue)):
                x,y=queue.pop()
                neighbors=[(x+addX,y+addY) for addX,addY in dirs]
                if self.logging: print("current",x,y)
                if self.logging: print("neighbors",neighbors)
                for neighbor in neighbors:
                    nextX,nextY=neighbor
                    if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and grid[nextX][nextY]==1 and  neighbor not in queue:
                        queue.insert(0,neighbor)
                        
                if grid[x][y]!=2:       ### I made a mistake here by not checking and simply subtracting
                    grid[x][y]=2
                    freshCount-=1
                if self.logging: print("queue",queue)
                        
            level+=1       
            
        if freshCount>0:
            return -1
        else:
            return level-1      ## READ LEVELS THEORY. THIS PLACE RETURNS 1,2,3
---------------------------------------------------------------------
#VISITED MARKING BEFORE POP -- I dont use this

class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # if grid[0][0]==1 and len(grid[]):
        #     return 0
        
        freshCount=0
        queue= []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==2:
                    queue.append([i,j])
                elif grid[i][j]==1:
                    freshCount+=1
        
        if freshCount==0:return 0   ##### this is important
        
        directions=[[1,0],[-1,0],[0,1],[0,-1]]
        level=0       
        while queue:       
            for i in range(len(queue)):
                
                current = queue.pop() 
                neighborlist= [[x[0]+current[0],x[1]+current[1]] for x in directions]
                for neighbor in neighborlist:
                    if neighbor[0]>=0 and neighbor[0]<len(grid) and neighbor[1]>=0 and neighbor[1]<len(grid[0]) and grid[neighbor[0]][neighbor[1]]==1:
                        queue.insert(0,neighbor)
                        grid[neighbor[0]][neighbor[1]]=2
                        freshCount-=1
                        if freshCount==0: return level+1    ##### marking before adding to queue IMP !!!!!
                        #### why because we are at level+1 already but it increaes when the loop completes
                        ### We can do the visited operation after also.
                        ### That will also work. But two changes (only level) and have to do fresh counting properly  
                        
            level+=1
    
        return -1

======================================================================  
286. Walls and Gates
---------------------------------------------------------------------
BFS. Exactly same logic as the question before. rotting oranges. 
1. There is no need to keep track of NoOfRooms, if soom room cant be reached we have to store inf anyway which is the default value 
2. Are you thinking about collision of two BFS or overwriting or taking minimum at a point? You dont have to worry about this because only the first BFS reach is recorded and marked as level and its marked visited after that.  
3. CHECK QUEUE LAST THAT IS THE MOST EXPENSIVE CHECK. THIS WILL AVOID TLE
---------------------------------------------------------------------
## VISITED MARK AFTER POP
ALGO_NAME: GRAPH_STYLE_BFS_ITERATIVE

class Solution(object):
    logging=False
    def wallsAndGates(self, rooms):
        m=len(rooms)
        n=len(rooms[0])
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        queue=[]
        for i in range(m):
            for j in range(n):
                if rooms[i][j]==0:
                    queue.insert(0,(i,j))
        
        level=0
        while queue:
            for i in range(len(queue)):
                x,y=queue.pop()
                neighbors=[(x+addX,y+addY) for addX,addY in dirs]
                for neighbor in neighbors:
                    nextX,nextY=neighbor
                    if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and rooms[nextX][nextY]==2147483647 and neighbor not in queue:
                        queue.insert(0,neighbor)
                rooms[x][y]=level    
            level+=1 
        return rooms
---------------------------------------------------------------------
## VISITED MARK BEFORE POP  ##NOT RECOMMENDED
class Solution(object):
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: None Do not return anything, modify rooms in-place instead.
        """
        queue=[]
        noOfRooms=0
        for i in range(len(rooms)):
            for j in range(len(rooms[0])):
                if rooms[i][j]==0:
                    queue.append([i,j])
                elif rooms[i][j]==2147483647:
                    noOfRooms+=1
        
        direction=[[1,0],[-1,0],[0,1],[0,-1]]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                neighbors=[[current[0]+dir1[0],current[1]+dir1[1]] for dir1 in direction]
                for x in neighbors:
                    nextX=x[0]
                    nextY=x[1]
                    if nextX>=0 and nextX<=len(rooms)-1 and nextY>=0 and nextY<=len(rooms[0])-1 and rooms[nextX][nextY]==2147483647:
                        rooms[nextX][nextY]=level+1        ##### marking before adding to queue IMP !!!!!
                        #### why because we are at level+1 already but it increaes when the loop completes
                        ### We can do the visited operation after also.
                        ### That will also work. But two changes (only level) and have to do fresh counting properly
                        queue.insert(0,[nextX,nextY])
                        noOfRooms-=1
                        if noOfRooms==0: return
                        
            level+=1
        

======================================================================

102. Binary Tree Level Order Traversal
Given a binary tree, return the level order traversal of its nodes' values. 
(ie, from left to right, level by level).
---------------------------------------------------------------------
Code: ALGO_NAME: TREE_STYLE_BFS_ITERATIVE  
Done
---------------------------------------------------------------------

class Solution(object):
    def levelOrder(self, root):
        if not root: ##edge case not base case
            return []
        
        ret=[]
        queue=[root]
        while queue:
            currLevel=[]
            for i in range(len(queue)):    #### this is the way we count levels. why does this work because at each     ##############################level we know the number of members of next level. 
                current=queue.pop()
                neighbors=[current.left,current.right]
                for neighbor in neighbors:   #### ORDER IS IMPORTANT ## LEFT IS GOING IN FIRST SO LEFT WILL COME OUT FIRST (FIFO IN QUEUE)
                    if neighbor:
                        queue.insert(0,neighbor)
                currLevel.append(current.val)
            ret.append(currLevel)
        return ret
======================================================================
314. Binary Tree Vertical Order Traversal
Requires a level order traversal /bfs
This is a modified version of level order traversal(BFS) here we deal with left right separately, also we have a column variable we keep track of add to a dictionary.
DFS wont work because the other of common columns is getting mixed up

ALGO_NAME: TREE_STYLE_BFS_ITERATIVE
class Solution(object):
    def verticalOrder(self, root):
        if not root:
            return 
        
        column=0
        queue=[(root,column)]
        dict1={}
        
        while queue:
            for i in range(len(queue)):
                current,column=queue.pop()
                if current.left:
                    queue.insert(0,(current.left,column-1))
                if current.right:
                    queue.insert(0,(current.right,column+1))
                        
                if column not in dict1:
                    dict1[column]=[current.val]
                else:
                    dict1[column].append(current.val)
        
        ans=[]
        list1=sorted(dict1.keys())
        for i in range(len(list1)):
            ans.append(dict1[list1[i]])    
        return ans
======================================================================
543. Diameter of Binary Tree
question tricks me up. TRICKY TRICKY REMEMBER
wow this is also possible my dfs and is the basis for another much more difficult question in graph!

https://medium.com/@tbadr/tree-diameter-why-does-two-bfs-solution-work-b17ed71d2881


ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION 

class Solution(object):
    def diameterOfBinaryTreeR(self,root):
        if not root:
            return (0,0)        ### EVEN THIS IS IMPORTANT ### TRICKY TO REALIZE
        if not root.left and not root.right:  ##leaf node
            return (0,1)
        maxDiameterLeft,heightLeft =self.diameterOfBinaryTreeR(root.left)
        maxDiameterRight,heightRight=self.diameterOfBinaryTreeR(root.right)
        maxDiameterThisNode=max(maxDiameterLeft,maxDiameterRight,heightLeft+heightRight)
        maxheightThisNode=max(heightLeft,heightRight)+1
        
        return maxDiameterThisNode,maxheightThisNode
                
    def diameterOfBinaryTree(self, root):
        return self.diameterOfBinaryTreeR(root)[0]
        
### more inefficient way of using depth check

class Solution(object):
    def maxTreeDepth(self,root):
        if not root:
            return 0
        queue=[root]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                for x in [current.left,current.right]:
                    if x:
                        queue.insert(0,x)
            level+=1
        return level
                
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if not root.left and not root.right:  ##leaf node
            return 0
        x=self.diameterOfBinaryTree(root.left)
        y=self.diameterOfBinaryTree(root.right)
        ansFromThisNode= self.maxTreeDepth(root.left)+self.maxTreeDepth(root.right)
        return max(x,y,ansFromThisNode)
        #return self.maxTreeDepth(root)
---------------------------------------------------------------------
Diameter of Binary Tree/

The key observation to make is: the longest path has to be between two leaf nodes

Diameter of Binary Tree
1. Use ACROBATIC 2 in graphs. Get max height of left subtree and right subtree using BFS. 
 max(heightLeft,heightRight)+1

======================================================================
Directed one, bfs will be pain so use head recursion. 

/1522. Diameter of N-Ary Tree
https://leetcode.com/problems/diameter-of-n-ary-tree/discuss/1578828/Java-Simple-and-Clean-solution-w-Explanation-or-Beats-100-0ms-or-TC%3A-O(M)-SC%3A-O(H)

annoying question, used head recursion like binary directed version. 

---------------------------------------------------------------------
I used this logic, either the max dia passes through the root or it doesnt. 
If it does: max dia = sum of top 2 child HEIGHTS. like before in binary tree.
If it doesnt: max dia = 


---------------------------------------------------------------------
Length of the longest path passing through a node = 
1st Max Height among N children + 2nd Max Height among N children


----------------------------------------------------------------------
class Solution(object):
    logging=True
    def diameterR(self, root):
        if not root.children:        ## base case here is not null but leaf node
            return (1,0)
        
        
        root_ans=[]
        max_child_dia=float("-inf")
        
        for child in root.children:
            child_height,child_dia = self.diameterR(child)
            max_child_dia = max(max_child_dia,child_dia)
            root_ans.append((child_height,child_dia))
        if self.logging: print("child ans list for",root.val ,root_ans)
        
        root_ans.sort(key=lambda x: x[0],reverse=True) ## get the top 2 child heights
        root_dia = sum([x[0] for x in root_ans[:2]])   ## sum it up to get diameter passing through root
            
        max_root_dia = max(root_dia,max_child_dia)     ## but it is possible that largest dia doesnt pass through root
        max_root_height = max([x[0] for x in root_ans])+1 ## get max child height and add 1
        print("for root",root.val,"max_root_height",max_root_height,"max_root_dia",max_root_dia)
        return (max_root_height,max_root_dia)
    
    
    def diameter(self, root):
        return self.diameterR(root)[1]
======================================================================
1245. Tree Diameter
UNDIRECTED HERE!!! NOW EVEN WHEN ITS A "TREE" you have to do graph style bfs.

https://medium.com/@tbadr/tree-diameter-why-does-two-bfs-solution-work-b17ed71d2881
find the farthest point A by running bfs on any node, from the leaf run bfs again and find the farthest B.
Return A and B as the endpoints of the tree diameter.


import collections
class Solution(object):
    logging=False
    def bfs(self,node):
        queue=[node]
        level=0
        visited={}
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                neighbors=self.adjacencyList[current]
                if self.logging: print("current",current)
                if self.logging: print("neighbors",neighbors)
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in queue:
                        queue.insert(0,neighbor)
                visited[current]=1
            level+=1
        #if self.logging: print("current",current)
        return current,level-1 
            
    def treeDiameter(self, edges):
        self.adjacencyList=defaultdict(list)
        for edge in edges:
            self.adjacencyList[edge[0]].append(edge[1])
            self.adjacencyList[edge[1]].append(edge[0])
            
        #print(self.adjacencyList)  
        A=self.bfs(0)[0]
        if self.logging: print("A",A)
        return self.bfs(A)[1]
======================================================================
310. Minimum Height Trees
v difficult question, lot of extra theory


1. Brute Force

n=4 
[[1,0],[1,2],[1,3]]

longest path - bfs and search for leaves? Yes
longest path - dfs and search for leaves?

repeated work
Leave nodes seem to be useless. 

Recursion ?
1+(max(1,1)+1)
(max(1,1),1)+1

O(N2) because we have to repeat this from every node.
Every node BFS is a O(N) operation. Here there are no repeat edges.

2. optimised solution
find the farthest point A by running bfs on any node, from the leaf run bfs again and find the farthest B.
A and B are the endpoints of the tree diameter. Center of this diameter is my answer. 
I would need an extra list to carry the path while I am coming back. 

method 1
============
# TREE STYLE OR GRAPH STYLE?? GRAPH STYLE because undirected
Why GRAPH STYLE? WHY NOT TREE STYLE? TREE STYLE ONLY IN DIRECTED TREES OTHERWISE GRAPH STYLE!!


Graph style is giving me pain because of list in the state
I have never done Graph style with lists in the STATE!!!
All permutation, combination are tree styles. I didnt add the list in the visited!! This is a standard and I will remember this.


import collections
class Solution(object):
    logging=False
    def bfs(self,node):
        queue=[(node,[node])]
        level=0
        visited={}
        while queue:
            for i in range(len(queue)):
                current,path=queue.pop()
                neighbors=[(x,path+[x]) for x in self.adjacencyList[current] if x not in visited]
                if self.logging: print("current",current)
                if self.logging: print("neighbors",neighbors)
                for neighbor in neighbors:
                    nextCurrent,nextPath=neighbor
                    if nextCurrent not in visited: # and neighbor not in queue: I had to remove queue check part to pass
                                                   # remember queue check takes a lot of time in bfs/dfs
                                                   # 
                        queue.insert(0,neighbor)
                visited[current]=1
            level+=1
        #if self.logging: print("current",current)
        return current,path         ## once we complete we are at the farthest point
    
    
    def findMinHeightTrees(self, n, edges):
        self.adjacencyList=defaultdict(list)
        
        for edge in edges:
            self.adjacencyList[edge[0]].append(edge[1])
            self.adjacencyList[edge[1]].append(edge[0])
            
        A=self.bfs(0)[0]
        
        path=self.bfs(A)[1]
        ## we return the centroid of the path in case of odd or even
        if len(path)%2!=0:
            return [path[len(path)/2]]       
        else:
            return [path[len(path)/2-1],path[len(path)/2]]   
            
method2: BFS topsort. didnt do this. 

======================================================================
987. Vertical Order Traversal of a Binary Tree
Same as before but more annoying because now we have to keep track of row too
ONLY BFS will give correct rows.
## The issue in this question is when they do match in column number, it cries about order of output.

From test case, the real requirement is:
If two nodes have the same position,

check the layer, the node on higher level(close to root) goes first
if they also in the same level, order from small to large


ALGO_NAME: TREE_STYLE_BFS_ITERATIVE
class Solution(object):
    def verticalTraversal(self, root):
        if not root:
            return 
        
        column=0
        row=0
        queue=[(root,row,column)]
        dict1={}
        
        while queue:
            for i in range(len(queue)):
                current,row,column=queue.pop()
                if current.left:
                    queue.insert(0,(current.left,row+1,column-1))
                if current.right:
                    queue.insert(0,(current.right,row+1,column+1))
                        
                if column not in dict1:
                    dict1[column]=[[current.val,row]]
                else:
                    dict1[column].append([current.val,row])
        
        ans=[]
        list1=sorted(dict1.keys()) ### early columns come first
        for i in range(len(list1)):
            dict1[list1[i]].sort(key=lambda x: (x[1],x[0])) ## first row, then value  ## headache
            ans.append([x[0] for x in dict1[list1[i]]])  ## extract value 
        return ans
======================================================================
1091. Shortest Path in Binary Matrix
This is a BFS not a DP why?
Because for a DP you need to calculate. the one right up and left. but here each depends on the other. So DP is impossible.
There is mutual dependency among subproblems

ALGO_NAME: GRAPH_STYLE_BFS_ITERATIVE
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        n=len(grid)
        dirs=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
        queue=[[0,0]]
        level=0
        if grid[0][0]!=0:
            return -1
        
        while queue:
            for i in range(len(queue)):
                x,y=queue.pop()
                neighbors=[(x+addX,y+addY) for addX,addY in dirs]
                for neighbor in neighbors:
                    nextX,nextY=neighbor
                    if nextX>=0 and nextX<=n-1 and nextY>=0 and nextY<=n-1 and grid[nextX][nextY]==0 and neighbor not in queue:
                        queue.insert(0,neighbor)
                grid[x][y]="*"
                if (x,y)==(n-1,n-1):
                    return level+1 ##0,1,2 returned here but i need 1,2,3 THEORY OF levels
            level+=1
        return -1               ### doesnt return level earlier so unreachable
    
======================================================================
200. Number of Islands
==============================
Similar to connect components or Friend circles 
Starting BFS/DFS at every point. and marking whole connected graph as visited
==============================
ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK

class Solution(object):
    def dfs(self,i,j,grid):
        m=len(grid)
        n=len(grid[0])
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        stack=[(i,j)]
        #visited={}
        while stack:
            x,y=stack.pop()
            neighbors=[(x+addX,y+addY) for addX,addY in dirs]
            for neighbor in neighbors:
                nextX,nextY=neighbor
                if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and grid[nextX][nextY]=="1" and neighbor not in stack:
                    stack.append(neighbor)
            grid[x][y]="0"
    
    def numIslands(self, grid):
        m=len(grid)
        n=len(grid[0])
        count=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]=="1":
                    count+=1
                    self.dfs(i,j,grid)
        return count
                
---------------------------------------------------------------------
Marking visited before -- I dont use this
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        count=0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=='1':
                    queue=[[i,j]]
                    directions=[[1,0],[-1,0],[0,1],[0,-1]]
                    while queue:
                        current=queue.pop()
                        neighbors=[[current[0]+dir1[0],current[1]+dir1[1]] for dir1 in directions ]
                        for x in neighbors:
                            nextX=x[0]
                            nextY=x[1]
                            if nextX>=0 and nextX<=len(grid)-1 and nextY>=0 and nextY<=len(grid[0])-1 and grid[nextX][nextY]=='1':
                                queue.insert(0,[nextX,nextY])
                                grid[nextX][nextY]='0' 
                                 ##### marking before adding to queue IMP !!!!!
		                        #### why because we are at level+1 already but it increaes when the loop completes
		                        ### We can do the visited operation after also.
		                        ### But that will have more time complexity as it will allow more duplicates to be added to the queue
                            
                    count+=1 
        
        return count 
====================================================================== 
695. Max Area of Island
Same as number of islands, we just keep count now in each unconnect component and return max

ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK

class Solution(object):
    def dfs(self,i,j,grid):
        count=0
        m=len(grid)
        n=len(grid[0])
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        stack=[(i,j)]
        #visited={}
        while stack:
            x,y=stack.pop()
            neighbors=[(x+addX,y+addY) for addX,addY in dirs]
            for neighbor in neighbors:
                nextX,nextY=neighbor
                if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and grid[nextX][nextY]==1 and neighbor not in stack:
                    stack.append(neighbor)
            grid[x][y]=0
            count+=1 
        return count
    def maxAreaOfIsland(self, grid):
        m=len(grid)
        n=len(grid[0])
        maxCount=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    count=self.dfs(i,j,grid)
                    maxCount=max(maxCount,count)
        return maxCount
======================================================================  
419. Battleships in a Board
Exactly as number of islands, we count the number of unconnected components
ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK

class Solution(object):
    def dfs(self,i,j,board):
        m=len(board)
        n=len(board[0])
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        stack=[(i,j)]
        #visited={}
        while stack:
            x,y=stack.pop()
            neighbors=[(x+addX,y+addY) for addX,addY in dirs]
            for neighbor in neighbors:
                nextX,nextY=neighbor
                if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and board[nextX][nextY]=="X" and neighbor not in stack:
                    stack.append(neighbor)
            board[x][y]="."
    
    def countBattleships(self, board):
        m=len(board)
        n=len(board[0])
        count=0
        for i in range(m):
            for j in range(n):
                if board[i][j]=="X":
                    count+=1
                    self.dfs(i,j,board)
        return count
---------------------------------------------------------------------
Optimized in follow up 
We dont really need DFS too. We can simply count starting/ending points and ending points of the ships.
Starting points are simply like this
xxxxxx  or x
           x
           x
           x

Edge case: x 4 sides of water, count this as 2
What does this mean? 3 sides are water at any starting/ending point. We will consider out of board as water too.
now do this for every point. divide final count by 2

class Solution(object):    
    def countBattleships(self, board):
        dirs=[[1,0],[-1,0],[0,1],[0,-1]]
        count=0
        m = len(board)
        n = len(board[0])
        for i in range(m):
            for j in range(n):
                if board[i][j]=="X":
                    neigbors=[[i+dir1[0],j+dir1[1]] for dir1 in dirs]
                    water=0
                    for neighbor in neigbors:
                        nextX=neighbor[0]
                        nextY=neighbor[1]
                        if (nextX<0 or nextX>m-1 or nextY<0 or nextY>n-1) or board[nextX][nextY]==".":
                            water+=1
                    if water==3:
                        count+=1
                    elif water==4: ## two start and end points collapsed as one
                        count+=2
        return count/2
======================================================================                          
                 
107. Binary Tree Level Order Traversal II
---------------------------------------------------------------------
Code: Simple BFS solution, CORE BFS logic . Same as the last question. just bottom up now. 
Done
---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_BFS_ITERATIVE
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:  ##edge case
            return []
        
        ret=[]
        while queue:
            curLevel=[]
            for i in range(len(queue)):
                current=queue.pop()
                for x in [current.left,current.right]:
                    if x:
                        queue.insert(0,x)
                curLevel.append(current.val)
            ret.insert(0,curLevel) #or append and then reverse it
        return ret  

======================================================================

637. Average of Levels in Binary Tree
---------------------------------------------------------------------
Code: Simple BFS solution, CORE BFS logic . Same as the last question. just consider float thing
Done
---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_BFS_ITERATIVE
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root:  ##edge case
            return []
        queue=[root]
        ret=[]
        while queue:
            curLevel=[]
            for i in range(len(queue)):
                current=queue.pop()
                for x in [current.left,current.right]:   
                    if x:
                        queue.insert(0,x)
                curLevel.append(current.val)
            ret.append(curLevel)
        
        return [float(sum(x))/len(x) for x in ret] ### coverting to float is key 
======================================================================
199. Binary Tree Right Side View

---------------------------------------------------------------------
Done,ALGO_NAME: TREE_STYLE_BFS_ITERATIVE
------------------------------------
class Solution(object):
    def rightSideView(self, root):
        if root:
            queue=[root]
        else:
            return 
        ret=[]
        level=0
        while queue:
            curLevel=[]
            for i in range(len(queue)):
                current=queue.pop()
                neighbors=[current.left,current.right]
                for neighbor in neighbors:  #### ORDER IS IMPORTANT ## LEFT IS GOING IN FIRST SO LEFT WILL COME OUT FIRST (FIFO IN QUEUE)
                    if neighbor:
                        queue.insert(0,neighbor)
                curLevel.append(current.val)
            ret.append(curLevel)
            level+=1
        return [row[-1] for row in ret] 
======================================================================        
 116. Populating Next Right Pointers in Each Node I and II
 ---------------------------------------------------------------------
Done, just bfs but start from the right side!!!!
PREV IS NONE IS SET INSIDE!!!!! THIS EFFECTIVELY RESETS PREV AT EVERY LEVEL MAKING IT NEVER POINT TO THE PARENT!!!
My answer doesnt change if the tree is perfect or not. So same for II
---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_BFS_ITERATIVE

class Solution(object):
    def connect(self, root):
        if not root:   ##edge case 
            return None
        
        while queue:
            prev=None
            for i in range(len(queue)):
                current=queue.pop()
                for x in [current.right,current.left]: #FIFO
                    if x:
                        queue.insert(0,x)
                current.next=prev
                prev=current
        return root  ### dont forget to return even if it is in place
---------------------------------------------------------------------
SAFER WAY IS LIKE THIS where I add all level nodes to a list
class Solution(object):
    def connect(self, root):
        if root:
            queue=[root]
        else:
            return 
        ret=[]
        level=0
        while queue:
            curLevel=[]
            for i in range(len(queue)):
                current=queue.pop()
                neighbors=[current.left,current.right]
                for neighbor in neighbors:  #### ORDER IS IMPORTANT ## LEFT IS GOING IN FIRST SO LEFT WILL COME OUT FIRST (FIFO IN QUEUE)
                    if neighbor:
                        queue.insert(0,neighbor)
                curLevel.append(current)
            ret.append(curLevel)
            level+=1
    
        for i in range(len(ret)):
            for j in range(len(ret[i])-1):
                ret[i][j].next=ret[i][j+1]
        return root

======================================================================
101. Symmetric Tree
HEAD RECURSION IS VERY Tricky!
---------------------------------------------------------------------
Done both
just bfs and checking levels but I have to include the NULLs during the bfs to compare 
NULLS ARE ADDED TO STACK AND AFTER POPPING,NEIGHBORS arent checked
or 
RECURSION - 
It is symmetric if left and right trees are mirrors!!!
mirror has 3 rules, left side and right side should be mirrors not equal, roots should be same
RECURSION IS VERY VERY TRICKY
---------------------------------------------------------------------
ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION

class Solution(object):
    def isMirror(self, root1,root2):
        if not root1 and  not root2:   ## base cases 
            return True
        elif not root1 or not root2:
            return False

        return self.isMirror(root1.left,root2.right) and self.isMirror(root1.right,root2.left) and root1.val==root2.val
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """    
        return self.isMirror(root.left,root.right)

---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_BFS_ITERATIVE

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        if root:
            queue=[root]
        else:
            return True
        
        while queue:
            curLevel=[]
            for i in range(len(queue)):
                current=queue.pop()
                if current:
                    for x in [current.right,current.left]:
                        queue.insert(0,x) ###insertion without checking if NULL
                    curLevel.append(current.val)
                else:
                    curLevel.append(None)
            if curLevel!=curLevel[::-1]:
                return False
        return True
======================================================================
111. Minimum Depth of Binary Tree
---------------------------------------------------------------------
Done , 
just bfs and checking levels but I have to check for leaf nodes and report level if i find them.
Mistake checking for first None and then reporting level
---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_BFS_ITERATIVE

class Solution(object):
    def minDepth(self, root):
        if not root:
            return 0
        
        queue=[root]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                for x in [current.left,current.right]:
                    if x:
                        queue.insert(0,x)
                if current.left==None and current.right==None:
                    return level+1 #### why because we are at level+1 already but it increaes when the loop completes  
            level+=1  
---------------------------------------------------------------------
TRICKY DUE TO SKEWED CASES

class Solution(object):
    def minDepth(self, root):
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
    

        if root.left:
            depthL=self.minDepth(root.left)
        else:
            depthL=float("inf")
        if root.right:
            depthR=self.minDepth(root.right)
        else:
            depthR=float("inf")
        
        return min(depthL,depthR)+1

======================================================================              
104. Maximum Depth of Binary Tree
---------------------------------------------------------------------
Done, basic bfs
---------------------------------------------------------------------
just plain bfs with no changes at ALL !!
ALGO_NAME: TREE_STYLE_BFS_ITERATIVE

class Solution(object):
    def maxDepth(self, root):
         if not root:
            return 0
        
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                for x in [current.left,current.right]:
                    if x:
                        queue.insert(0,x)
            level+=1
        return level
======================================================================
559. Maximum Depth of N-ary Tree
---------------------------------------------------------------------
Done, ALGO_NAME: TREE_STYLE_BFS_ITERATIVE
--------------------------------------------------------------------- 
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        
         if not root:
            return 0
        
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                for x in current.children:
                    if x:
                        queue.insert(0,x)
            level+=1
        return level
======================================================================
133. Clone Graph
---------------------------------------------------------------------
Done
We need to clone each and every point. through dfs, bfs whatever
But after cloning we need to give neighbors as the new nodes and not the old ones. We use visited carefully for this
for keeping a map of the cloned Nodes as adjacency List isnt given

The neighbors in copied nodes still point to original nodes. so i need map of orig:copied

1. Went to visited, got an original and its corresponding neighbors (using original)
Now since I have the copied this visited, I modify the copied node neighbors by iterating on ORIGINAL neighbors.
IF YOU TRY TO ITERATE ON THE NEIGHBORS OF THE COPIED THAT WILL GIVE ERROR?
1. You are modifying the list while iterating on it! 
---------------------------------------------------------------------
## VISITED MARK AFTER POP  ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK
class Solution(object):
    def cloneGraph(self, node):
        ## adjacency list isnt needed because AL is needed for neighbors and thats given 
        
        if not node:            ##edge case
            return None
        
        visited={}
        stack=[node]
        while stack:
            current=stack.pop()
            copyNode=Node(current.val,[]) ## no point of copying neighbors as they will be from existing graph
            visited[current]=copyNode
            for neighbor in current.neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
            
        
        for origNode in visited:     
            visited[origNode].neighbors=[visited[x] for x in origNode.neighbors]
              
        return visited[node]
        

---------------------------------------------------------------------
class Solution:   
    def cloneGraph(self, node):
        if not node:
            return 
        
        visited={}
        queue=[node]
        while queue:
            current=queue.pop()
            for x in current.neighbors:
                if x not in visited:      ##### dont forget this 
                    queue.insert(0,x)
            
            copiedNode=Node(val=current.val,neighbors=[])
            visited[current]=copiedNode
        
        for origNode in visited:
            visited[origNode].neighbors= [visited[x] for x in origNode.neighbors]
    
        return visited[node] 

Exactly same logic : 138. Copy List with Random Pointer       
======================================================================
103. Binary Tree Zigzag Level Order Traversal
---------------------------------------------------------------------
same old BFS but calculate level and reverse each time or (just append and reverse later by index in list )
---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_BFS_ITERATIVE

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        ret=[]
        queue=[root]
        while queue:
            currLevel=[]
            for i in range(len(queue)):
                current=queue.pop()
                for node in [current.left,current.right]: ##FIFO ORDER IMPORTANT
                    if node:
                        queue.insert(0,node)
                currLevel.append(current.val)
            ret.append(currLevel)

        for i in range(len(ret)):
            if i%2!=0:
                ret[i]=ret[i][::-1]
        return ret
---------------------------------------------------------------------           
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        
        if not root:
            return 
        
        queue=[root]
        level=0
        ret=[]
        while queue:
            curLev=[]
            for i in range(len(queue)):
                current= queue.pop()
                for x in [current.left,current.right]:
                    if x:
                        queue.insert(0,x)
                curLev.append(current.val)
            
            if level%2==0:
                ret.append(curLev)
            else:
                ret.append(curLev[::-1])
            level+=1
            
        
        return ret
======================================================================          
127. Word Ladder
Logic : Its a bfs! Why? they are asking about levels. 
1. For each word in the given word list create an adjacency list
2. Now start bfs and count the levels to reach the target word.
3. Only add words to the stack which are not visited (bfs  logic ofcourse) and those that exist in the word list 
4. Convert wordlist to wordListDict will solve time exceeded issues and speeden up code!!!
---------------------------------------------------------------------           
ALGO_NAME: GRAPH_STYLE_BFS_ITERATIVE

class Solution(object):
    def generateNeighbors(self,word,wordListDict):
        letters="abcdefghijklmnopqrstuvwxyz"
        listOfNeighbors=[]
        for i in range(len(word)):
            for letter in letters:
                newWord=word[:i]+letter+word[i+1:]
                if newWord in wordListDict:
                    listOfNeighbors.append(newWord)
        return listOfNeighbors
        
    def ladderLength(self, beginWord, endWord, wordList):
        wordListDict={}
        for word in wordList:
            if word not in wordListDict:
                wordListDict[word]=1
        visited={}
        queue=[beginWord]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                for word in self.generateNeighbors(current,wordListDict):
                    if word not in queue and word not in visited:
                        queue.insert(0,word)
                visited[current]=1
                if current==endWord:
                    return level+1  
                    ### level down and you want to return 1,2,3 instead of 0,1,2

            level+=1
        return 0
---------------------------------------------------------------------           
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        wordList=wordList+[beginWord]. ## note how begin word is not included ##not needed
        
        dict1={}
        for i in range(len(wordList)):
            dict1[wordList[i]]=[]
        
        for j in dict1:
            dict1[j]=self.create_adjacent_words(j,dict1)
        
        queue=[beginWord]
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                if current==endWord:  ###### not how i end 
                    return level+1 #### why because we are at level+1 already but it increaes when the loop completes
                for x in dict1[current]:
                    if x!='visited':     #### removing visited nodes
                        queue.insert(0,x)
                dict1[current]=['visited']    #### Here i used the same adjacency matrix but you can create "visited map and mark visited before adding to the queue"
            level+=1
            
        return 0 
            
        
    def create_adjacent_words(self,word,dict1):
        letters=str('abcdefghijklmnopqrstuvwxyz')
        list_of_words=[]
        for i in range(len(word)):
            for letter in letters:
                if letter!=word[i]: ### this line can be easily missed
                    new_word=word[:i]+str(letter)+word[i+1:]
                    if new_word in dict1:
                        list_of_words.append(new_word)
    
        return list_of_words
        

110. Balanced Binary Tree
Tags: Recursion, BFS 
======================================================================
This question is about recursion and also about BFS 
Method1:  O (N2)
1. I am confused so I simply call my depth calculator (using BFS) at each left and right node. This seems like repeat calculation to me. But I guess for Brute force its ok ??
2. Struture of recursion a) Two functions needed ? No. No need of carrying extra parameters.
                         b) See the Structure of recursion. We are going down the recursion stacks and we can only return True at the end at the base case. Remember this structure. 
                         c) Nature of recursion:
							Tail nature    --  it says false at each step while going down. Finds True only at leaf of recursion. 
							Non-tail nature-- while coming back up we aggreate the answer found at the end of each leaf. 


Method2: See in Recursion. Faster. O(N)

======================================================================
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root==None:            #### base case 
            return True
        
        if abs(self.maxDepth(root.left)-self.maxDepth(root.right))>1:    ### processing logic 
            return False
        
        return self.isBalanced(root.left) and self.isBalanced(root.right)  ## calling recursion again 
    
  ##  [1,2,2,3,null,null,3,4,null,null,4]
        
        
    def maxDepth(self,root):
        if root:
            queue=[root]
        else:
            return 0
        level=0
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                for x in [current.left,current.right]:
                    if x:
                        queue.insert(0,x)
            level+=1
        return level     
======================================================================
863. All Nodes Distance K in Binary Tree
First convert the tree to graph, create adjacency list, (dfs)
second, perform bfs on the adjacency list graph to find level k elements 

ALGO_NAME: TREE_STYLE_BFS_ITERATIVE
ALGO_NAME: GRAPH_STYLE_DFS_ITERATIVE_BACKTRACK



class Solution(object):
    def distanceK(self, root, target, k):
        adjacencyList={root.val:[]}
        
        stack=[root]
        while stack:
            current=stack.pop()
            for neighbor in [current.right,current.left]:
                if neighbor:
                    adjacencyList[neighbor.val]=[current.val]
                    stack.append(neighbor)
                    #print(neighbor.val, "appeneded")
            if current.left:
                adjacencyList[current.val].append(current.left.val)
            if current.right:
                adjacencyList[current.val].append(current.right.val)
        if not adjacencyList: return []
        #print(adjacencyList)
        visited={}
        queue=[target.val]
        level=0
        ans=[]
        while queue:
            for i in range(len(queue)):
                current=queue.pop()
                for neighbor in adjacencyList[current]:
                    if neighbor not in visited:
                        queue.insert(0,neighbor)
                visited[current]=1
                # print("level",level)
                # print(current)
                if level==k: ans.append(current)
            level+=1
        return ans      

======================================================================
105. Construct Binary Tree from Preorder and Inorder Traversal
## we know elements are unique, thats this is possible

## we use head recursion and assume we know the answer to the recurse calls
## preorder list gives me the root node
## inorder list gives me HOW MANY ELEMENTS TO THE LEFT OF root node
## so using this I can partition both lists 

ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder and not inorder:        ## other base cases arent possible
            return None                         ## leaf nodes have to have Null as their child
            
        
        root=TreeNode(preorder[0])
        for i in range(len(inorder)):       ##iterate to find the value preorder[0]
            if inorder[i]==preorder[0]:
                break   ## got the value of i
            
        ## we know now that there are i nodes to the left of root, so we calculate left subtree and right subtree in both arrays
        root.left = self.buildTree(preorder[1:1+i],inorder[:i])
        root.right= self.buildTree(preorder[1+i:],inorder[i+1:])
        
        return root
Time Complexity : O(N2) where N is the number of nodes in the tree. At each recursion, we are choosing preorder[0] and searching it in inorder. This may take O(N) for each search and we are building a tree with N such nodes. So total time complexity is O(N2)
Space Complexity : O(N), required by implicit recursive stack. In worst case of skewed tree, the recursion depth might go upto N.
Important concept: even when there is no extra variable, the stack in the recursion is increasing the space complexity
--------------------------------------------------------------------------------------------------------------------------------------
Optimization: Use hashmap to store inorder
Now the problem with hashmap approach is indexing mostly. we have calculated the indexes in inorder with respect to the original inorder string.
Now if in your solution, you are passing modified inorder. ofcourse the indices arent valid anymore.
Solution? Carry the same inorder every time but pass two extra args from start& begin
For preorder, you can simply keep popping or increasing the indices as its one direction only and unit increase only.
Honestly the without hashmap approach is more intuitive and less confusing

ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION

class Solution(object):
    def buildTreeR(self,start,end,preorder,inorder):        ### here start and end are inclusive and keeping track of indices in inorder
        if start>end:
            return 
            
        root=TreeNode(preorder.pop(0))
        #root=TreeNode(preorder[self.preorder_index])   OR LIKE THIS
        #self.preorder_index+=1
        i=self.map[root.val] 
        root.left = self.buildTreeR(start,i-1,preorder,inorder)     ### ORDER OF LEFT AND RIGHT IS IMPORTANT HERE. WHY ?
                                                                    ### UNLIKE THE PREVIOUS QUESTION, WE ARENT KEEPING TRACK OF PREORDER 
                                                                    ## ARRAY'S INDICES VERY PROPERLY, JUST INCREASING SO LEFT SIDES 
                                                                    ## ROOT HAS TO COME FIRST
        root.right= self.buildTreeR(i+1,end,preorder,inorder)
        
        return root

    def buildTree(self, preorder, inorder):
        self.map={}
        #self.preorder_index=0
        for j in range(len(inorder)):
            self.map[inorder[j]]=j
            
        return self.buildTreeR(0,len(inorder)-1,preorder,inorder)
======================================================================
106. Construct Binary Tree from Inorder and Postorder Traversal
## ofcourse we will do it recursively like before
## first element of preorder gave root location 
## so last element of postorder will give root location

## POSTORDER is LeftRightRoot. The issue the separation between left and right
## Preorder is RootLeftRight. The issue the separation between left and right again. 
## Use number of nodes to left info from inorder to separate
ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not inorder and not postorder:
            return None
        
        root=TreeNode(postorder[-1])
        for i in range(len(inorder)):
            if inorder[i]==postorder[-1]:
                break       ## we got the index we are looking for
            
        
        root.left=self.buildTree(inorder[:i],postorder[:i])
        root.right=self.buildTree(inorder[i+1:],postorder[i:-1])
        return root
--------------------------------------------------------------------------------------------------------------------------------------
class Solution:
    def buildTreeR(self,start,end,inorder,postorder):        ### here start and end are inclusive and keeping track of indices in inorder
        if start>end:
            return 
            
        root=TreeNode(postorder.pop())
        #root=TreeNode(preorder[self.preorder_index])   OR LIKE THIS
        #self.preorder_index+=1
        i=self.map[root.val] 
        root.right= self.buildTreeR(i+1,end,inorder,postorder)          ### RIGHT MUST COME FIRST. WHY???
        root.left = self.buildTreeR(start,i-1,inorder,postorder) 
        
        
        return root

    def buildTree(self, inorder, postorder):
        self.map={}
        #self.preorder_index=0
        for j in range(len(inorder)):
            self.map[inorder[j]]=j
        return self.buildTreeR(0,len(inorder)-1,inorder,postorder)

        
======================================================================
889. Construct Binary Tree from Preorder and Postorder Traversal
## root, left , right
## left,right, root
Now the main issue is separation of left and right as before, but here in order isnt here to help us
I solved it myself by iterating on both and checking if we have reached equal nodes. did this by using 2 maps

ALGO_NAME: SIMPLE_TREE_STYLE_HEAD_RECURSION

class Solution(object):
    def constructFromPrePost(self, preorder, postorder):
        if not preorder and not postorder:
            return None
        
        root=TreeNode(preorder[0])        
        
        dict1={}
        dict2={}
        i=0
        while preorder[1:] and (len(dict1)==0 or dict1!=dict2):     ### had to deal with edge case of single value preorder and postorder 
            dict1[preorder[i+1]]=1
            dict2[postorder[i]]=1
            i+=1
        ### got i which is the number of nodes on left subtree ## rest is same after that 
        root.left=self.constructFromPrePost(preorder[1:i+1],postorder[:i])
        root.right=self.constructFromPrePost(preorder[i+1:],postorder[i:-1])

        
        return root     
## O(N2) time complexity

Now what i have used is that i have counted all nodes on left side to do the separation.
But we dont need to count all nodes, as the iteration rootLeftRight in the "Left" part too root comes first in preorder, 
so first element in the Left part is the root of the left subtree in preorder
and the last element in the Left part is the root of left subtree in postorder!! so simply take the preorder[1] and search for it in postorder.
That will mark the separation.
Now this is again O(N2) but as i search in the postorder array for prorder[1]. what if i have a hashmap for postorder then i dont need to search and
we have a O(N) solution
## withut hashmap
class Solution(object):
    def constructFromPrePost(self, preorder, postorder):
        if not preorder and not postorder:
            return None
        
        root=TreeNode(preorder[0])        
        
        for i in range(len(postorder)):
            if len(preorder)==1:
                i-=1
            elif postorder[i]==preorder[1]:       ### we search for preorder[1]
                break       ## we got the index we are looking for
                
        i=i+1
        #print("no of nodes",i)
        root.left=self.constructFromPrePost(preorder[1:i+1],postorder[:i])
        root.right=self.constructFromPrePost(preorder[i+1:],postorder[i:-1])
        return root  
## with hashmap will have deal with all the index stuff extra , do it later
======================================================================
++++++++++++++++++++++++++++++++++++++
+++++++Group : TOPOLOGIAL SORTING +++
++++++++++++++++++++++++++++++++++++++
Animation : https://www.youtube.com/watch?v=eL-KzMXSXXI 
Topological sorting mostly needs to understood and memorized, no real need to try 


https://leetcode.com/discuss/general-discussion/1078072/introduction-to-topological-sort
TOPOLOGICAL SORT -- DEFINED IN DIRECTED GRAPHS. 
This is simply a ordering of which node appears first and what appears later.LEAF TO ROOT.
These sorting are not unique. The leafs are not necessarily ALL at the end. its just each leaf comes before its own parent.


Not all directed graphs can have topo-sort, if it has a cycle then it wont exist-- it only exists in DAG(Directed Acyclic graph).
All trees are examples of DAGs(they dont have cycles by definition). It will have a topo-sort.

The key observation is that, leaf nodes should always come after their parents and ancestors. Following this intuition we can apply DFS and output nodes from leaves to the root.



MY TOPO SORTING goes from LEAF to ROOT, 
so 2 will come first.  
1->2


In this question either direction course to prereq or vice versa is fine. Just need to remember the direction of topo output and reverse accordingly.
207. Course Schedule
210. Course Schedule II
Test case: https://www.geeksforgeeks.org/topological-sorting/?ref=lbp _ from the diagram here. 
6
[[5,0],[4,0],[5,2],[4,1],[2,3],[3,1]]

PRINT FRIENDLY VERSION to UNDERSTAND WHATS HAPPENING
===================================
class Solution:
    def dfs_topo(self,current,visit,cycle,adjacencyList):
            if self.logging: print("cycle",cycle)
            if self.logging: print("visit",visit)
            if current in cycle:    ### CYCLE DETECTED
                if self.logging: print("cycle detected")
                return False
            if current in visit:    ### HAS BEEN ADDED TO OUTPUT
                if self.logging: print(current ,"has been visited before")
                return True
            
            cycle.add(current)
            for neighbor in adjacencyList[current]:
                if self.logging: print("enetering neighbor dfs",neighbor)
                if self.dfs_topo(neighbor,visit,cycle,adjacencyList) == False:
                    return False
            if self.logging: print("cycle",cycle)
            cycle.remove(current)
            visit.add(current)
            if self.logging: print("visit",visit)
            self.ans.append(current)
            if self.logging: print("self.ans",self.ans)
            return True
    
    def canFinish(self, numCourses, prerequisites): 
        self.logging=True
        vertexList=range(numCourses)
        adjacencyList = {vertex:[] for vertex in vertexList}
        for edge in prerequisites:                       
            adjacencyList[edge[0]].append(edge[1]) 
        print("adjacencyList",adjacencyList)
        self.ans = []
        visit, cycle = set(), set()
        
        for current in vertexList:
            print("+++++++++++++++++++++++++++++")
            print("current",current)
            if self.dfs_topo(current,visit,cycle,adjacencyList) == False:
                return []   ##I return emply list if ANY dfs top call returns False
        return self.ans 
--------------------------------------------------------------------------------------------------------------------------------------
Memorize this version of dfs_topo
1. Cycle check first
2. Visit set check second
3. We add current to the cycle and go run dfs_topo on neighbors. Before running we check if it returns False and return False if yes
4. We comes out of our exploration through dfs and remove current from cycle
5. We have visited current now so we add it to visit and to the answer. Here we made sure all children are visited before adding a parent. and at least starting from this leaf and downstream there was no loop

class Solution:
    def dfs_topo(self,current,visit,cycle,adjacencyList):
            if current in cycle:    ### CYCLE DETECTED
                return False
            if current in visit:    ### HAS BEEN ADDED TO OUTPUT
                return True
            
            cycle.add(current)
            for neighbor in adjacencyList[current]:
                if self.dfs_topo(neighbor,visit,cycle,adjacencyList) == False:
                    return False
            cycle.remove(current)
            visit.add(current)
            self.ans.append(current)
            return True
    
    def canFinish(self, numCourses, prerequisites): 
        vertexList=range(numCourses)
        adjacencyList = {vertex:[] for vertex in vertexList}
        for edge in prerequisites:                       
            adjacencyList[edge[0]].append(edge[1]) 
        ### vertex list and adjacency list created
        
        self.ans = []
        visit, cycle = set(), set()
        for current in vertexList:  ### RUN DFS_TOPO ON EVERY NODE, but if we detect cycle it will give False 
            if self.dfs_topo(current,visit,cycle,adjacencyList) == False:
                return False 
        if self.ans:
            return True 
======================================================================
1059. All Paths from Source Lead to Destination
We need to use the dfs_topo ofcourse. Why? this is Directed graph. We are asked to find if we can detect cycles in DAG. We know dfs_topo does this. 
Plus they are there is only ONE LEAF and thats destination. What is the condition of being head? no memebers in adjacency list. so check for that.

Now do we need dfs_topo from all nodes or just source? Why do we run it from every node. Because we need to sort it.
DFS_topo from a given node find its leafs. 
We run again and again because we might have started in the middle, or near the leafs

in this question its asking me to check:
    1. find leaf of source node and make sure leaf is destination. (now we know leaf has to exist. we just to check if its destination)
    2. No other leaves exist. Same as condition 1 single check
    3. no cycles present
Its not asking about leaves of other nodes, just source node. so dont run dfs_topo on others. There might be other leaf nodes or cycles present in the graph but thats ok bruh.


ALGO_NAME: DIRECTED_GRAPH_STYLE_DFS_TOPO

class Solution(object):
    def dfs_topo(self,current,visited,cycle,adjacencyList):
        if current in cycle:    ## Cycle detected return False
            if self.logging: print(current,"cycle detected")
            return False     
        if current in visited:  ## we have seen this 
            if self.logging: print(current,"visited")
            return True
        
        cycle.add(current)
        if not adjacencyList[current] and current!=self.destination:
            print(current)
            print(self.destination)
            if self.logging: print(current,"no adjacency but not dest")
            return False
            
        for neighbor in adjacencyList[current]:
            if self.dfs_topo(neighbor,visited,cycle,adjacencyList)==False:
                return False
        cycle.remove(current)
        visited.add(current)
        self.ans.append(current)
        return True
        
    def leadsToDestination(self, n, edges, source, destination):
        if not edges and n>1:
            return False
        if not edges and n<=1:
            return True
        
        self.source=source
        self.destination=destination
        edgeList=edges
        adjacencyList={}
        self.logging=False
        for edge in edgeList:
            if edge[0] in adjacencyList:
                adjacencyList[edge[0]].append(edge[1])  ### directed graph 
            else:
                adjacencyList[edge[0]]=[edge[1]]
            if edge[1] not in adjacencyList:
                adjacencyList[edge[1]]=[]
                
                
        if self.logging: print(adjacencyList)
        
        visited,cycle=set(),set()
        self.ans=[]
        
        if self.dfs_topo(source,visited,cycle,adjacencyList)==False: ###dfs_topo only from source not from all nodes
            return False
        return True
        
======================================================================
269. Alien Dictionary

### LOL THIS IS A TOPOLOGICAL GRAPH PROBLEM 
### what are the modes in the graph? chars which we see 
### is there a point in comparing words which differe more than one. No. we dont get any extra info. 
### now 
ALGO_NAME: DIRECTED_GRAPH_STYLE_DFS_TOPO

class Solution(object):
    logging=False
    def dfs_topo(self,current,visited,cycle,adjacencyList):
        if self.logging: print("cycle",cycle)
        if self.logging: print("current",current)
        #print(current in cycle)
        if current in cycle:
            if self.logging: print("current",current)
            if self.logging: print("cycle",cycle)
            if self.logging: print(current, "cycle detected")
            return False
        if current in visited:
            if self.logging: print(current, "visited")
            return True
        cycle.add(current)
        for neighbor in adjacencyList[current]:
            if self.logging: print("entering dfs of neighbor",neighbor)
            if self.dfs_topo(neighbor,visited,cycle,adjacencyList)==False:      ## i put current inside this by mistake
                return False
        cycle.remove(current)
        visited.add(current)
        self.ans.insert(0,current)
        if self.logging: print("ans",self.ans)
        return True                                                             ## i forget this

    def alienOrder(self, words):
        adjacencyList={}
        for word in words:
            for char in word:
                adjacencyList[char]=[]
        
        for i in range(0,len(words)-1,1):
            minL=min(len(words[i]),len(words[i+1]))
            if len(words[i])>len(words[i+1]) and words[i][:minL]==words[i+1][:minL]:
                return ""
            ##Cases word_i+1 is equal or bigger -- this is ok & natural
            ## smaller -- return "" if the prefix is same, handled above
            for j in range(minL):
                if words[i][j]!=words[i+1][j]:
                    if words[i+1][j] not in adjacencyList[words[i][j]]:     ## duplicate removal isnt really needed
                        adjacencyList[words[i][j]].append(words[i+1][j]) ##t->f
                    break                                               ##i forget to break after finding the letter
        if self.logging: print(adjacencyList)
        visited,cycle=set(),set()
        self.ans=[]
        for vertex in adjacencyList.keys():
            if self.logging: print("vertex",vertex)
            if self.dfs_topo(vertex,visited,cycle,adjacencyList)==False:
                return ""
        return "".join(self.ans)       
======================================================================
802. Find Eventual Safe States

Directed graph, "every possible path starting from that node leads to a terminal node."
This is ONLY POSSIBLE WHEN THERE IS NO CYCLE. WAIT DO I HAVE something for detecting cycle in cyclic graphs?
Yes ofcourse my beloved dfs_topo. So for each node run dfs_topo. But in my standard version I return False if any loops found at ANY nodes. Here if i find a loop i need to move on bruh. replace False with "continue". 

Now we need slightly deeper understanding of my code. At any node if we return after recursive dfs of neighbors and marking visited, this will only happen when there were no loops starting from this point and downstream. This node is safe. So anything which gets marked VISITED is safe. 
We could  have in effect started a dfs_topo on this from our base call but do i need to? no .I just mark it True on a global list. Now again there is no need to explicitly NOT call it again from base caller, because even if we do call this node has been marked visited anyway and we wont perform downstream neighbor calls again.So time complexity is still single DFS TOPO on entire graph time complexity. O(E+V).
At the end I only return my visited list back. Thats it

class Solution(object):
    logging=False
    def dfs_topo(self,current,visited,cycle,adjacencyList):
        #if self.logging: print("attemping dfs topo at",current)
        #if self.logging: print("cycle",cycle,"visited",visited)
        if current in cycle:
            if self.logging: print("returning cycle found in "+str(current))
            return False
        if current in visited:
            if self.logging: print("returning visited found in "+str(current))
            return True
        
        #if self.logging: print("cycle and visit passed,added "+str(current)+" to the cycle")
        cycle.add(current)
        neighbors=adjacencyList[current]
        for neighbor in neighbors:
            #if self.logging: print("starting dfs_topo on neigbors of "+str(current))
            if self.dfs_topo(neighbor,visited,cycle,adjacencyList)==False:
                return False
        cycle.remove(current)
        #if self.logging: print("removed "+str(current)+" from the cycle")
        visited.add(current)
        #print("marking ",current," as safe")
        self.safe[current]=True
        return True 
    
    
    def eventualSafeNodes(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[int]
        """
    
        visited,cycle=set(),set()
        for index in range(len(graph)):
            if self.logging: print("starting dfs topo at",index)
            if self.dfs_topo(index,visited,cycle,graph)==False: 
                continue
        return sorted(visited)
=====================================================================================
2115. Find All Possible Recipes from Given Supplies

# 1. Brute Force

# recipes = ["bread"] 
# ingredients = [["yeast","flour"]]
# supplies = ["yeast","flour","corn"]

# "bread"->yeast
# ->flour
# --------------------------------------
# ["bread","sandwich"]

# [["yeast","flour"],["bread","meat"]]

# ["yeast","flour","meat"]
# --------------------------------------
# ["bread","sandwich","burger"]
# [["yeast","flour"],["bread","meat"],["sandwich","meat","bread"]]

# ["yeast","flour","meat"] 
# --------------------------------------
At first thought you can think think of it as a simple iteration problem.
What makes it a graph problem. loops!

Directed graph
dfs_topo 
adjacency: [bread:["yeast","flour"],
Leaf nodes? anything which is not in recipes (but should be in supplies too)
So i just need to start dfs topo at each recipe and if I am able to visit all neighbors and come back and return True
we are good for that recipe

Decisions 
1. use index or string as current. I used string because you might encounter the string again. 
Will i go back and check all indices for string. No. I converted to dictionary.
2. reset cycle and visit for each recipe? treating as separate graphs? No not necessary. If you make it part of same graph, then repeated work is less. Things which have been completed are being added to visited to we dont go downstream again. 

            
class Solution(object):
    logging=False
    def dfs_topo(self,current,cycle,visit,recipes,ingredientsDict,supplies):
            if self.logging: print("cycle",cycle)
            if self.logging: print("visit",visit)
            if current in cycle:    ### CYCLE DETECTED
                if self.logging: print("cycle detected")
                return False
            if current in visit:    ### HAS BEEN ADDED TO OUTPUT
                if self.logging: print(current ,"has been visited before")
                return True
            
            cycle.add(current)
            
            if current in ingredientsDict:  ### i had to add this after an error because my leaf nodes arent in AL
                                            ### what happens for leaf nodes? AL is empty loop skips so did same.
                for neighbor in ingredientsDict[current]:
                    if neighbor not in supplies and neighbor not in ingredientsDict:
                        return False
                    if self.logging: print("enetering neighbor dfs",neighbor)
                    if self.dfs_topo(neighbor,cycle,visit,recipes,ingredientsDict,supplies) == False:
                        return False
                if self.logging: print("cycle",cycle)
            cycle.remove(current)
            visit.add(current)
            if self.logging: print("visit",visit)
            return True
    
    def findAllRecipes(self, recipes, ingredients, supplies):
        ingredientsDict={}
        for i in range(len(recipes)):
            ingredientsDict[recipes[i]]=ingredients[i]
        supplies=collections.Counter(supplies)
        
        ans=[]
        cycle,visit=set(),set()
        for recipe in recipes:      ## leaf nodes not included here and thats ok 
            if self.dfs_topo(recipe,cycle,visit,recipes,ingredientsDict,supplies)==True:
                ans.append(recipe)
            
        return ans   
=====================================================================================

++++++++++++++++++++++++++++++++++++++
+++++++Group : GRAPH ADVANCED +++
++++++++++++++++++++++++++++++++++++++
1584. Min Cost to Connect All Points

How many edges in a graph full connected n to n.
what do I want minimum n-1 edges?
Simply choosing minimum n-1 edges will not work. why because you can have an edge to a node which you are already connected to. whats the point of that useless edge

MST-- in order to find Minimum Spanning Tree we use Prims Algo:
1. Create adjacency list with all neighbors for every node and WEIGHTS for edges. A tuple of weight and node
2. start from any point, and lets look at all others, lets add all neighbors to a minHeap and pop the minimum. 
This way I am choosing greedily to include the next point into my connected graph. 
3. Now from that next point, lets add new unvisited stuff and their corresponding distance in the min heap. 
So now I have distances from 1 and distances from 2 to lets say a unconnected point 3. We get the minimum again. 
It can be from 1 or from 2 we dont care. This way we greedily add things to our connected graph. Once we add n-1 edges we are done or once we have nothing else to visit we are done. 





















++++++++++++++++++++++++++++++++++++++
+++++++Group : UNION FIND +++
++++++++++++++++++++++++++++++++++++++
If we have n nodes and n edges, we have to have a cycle for sure.
Union find by Rank 
In union find algo, we start by every node being parent of itself and rank 1. We perform merge opeartions again and again and assign parentage the to the node with bigger rank. at the end of it, we are able to find the cycle forming edges if any.

Union find 
Uses:
1. Detect Cycle in Non directed graph


ALGO_NAME: UNION_FIND

https://leetcode.com/discuss/general-discussion/1072418/Disjoint-Set-Union-(DSU)Union-Find-A-Complete-Guide




684. Redundant Connection

Union Find can be used to do cycle detection and elimination in undirected graph
class Solution:
    def findRedundantConnection(self, edges):
        parent = [i for i in range(len(edges) + 1)]  ### Initially each node is a parent of itself. 

        rank = [1] * (len(edges) + 1)               ### Everybody starts with a rank 1 
    
        def find(n):    ### for a given node find its root parent
            p = parent[n]
            while p != parent[p]:    ### we keep going up to the root, which has the property of being its own parent
                parent[p] = parent[parent[p]]  ## Path compression where we set the parent of p to its grand parent
                p = parent[p]                  ## reset p to its parent ##forgetting this line
            ### The loop is over when the root is found and we return the root node. 
            return p    
        
        # This union function returns False if already unioned
        
        def union(n1, n2):
            p1, p2 = find(n1), find(n2)     ### We find the root parents of n1 and n2
            
            if p1 == p2: return False       ### if parents are equal, they are already joined , we return False
            
                                            ## if parents are not same, we merge
            if rank[p1] > rank[p2]:         ### 
                parent[p2] = p1             ###p1 becomes parent of p2 because it has higher rank
                rank[p1] += rank[p2]        ###rank of p1 increases by rank p2
            else:                           
                parent[p1] = p2             ### vice versa    
                rank[p2] += rank[p1]
            return True                     ### we return that we were able to merge
    
        for n1, n2 in edges:                ### we go through all the edges, we should be able to do union mostly
            if not union(n1, n2):           ### except one case which is our answer
                return [n1, n2]
======================================================================

ALGO_NAME: UNION_FIND
261. Graph Valid Tree
## Basically cycle detection in an undirected graph
class Solution(object):
    logging=False
    def validTree(self, n, edges):
        if len(edges)<n-1:  ### have to make sure number of edges is enough, we need atleast n-1 edges to be a connected graph!!!!
            return False
        
        parent=[i for i in range(n)]
        rank=[1 for i in range(n)]
    
        def find(n):
            p=parent[n]
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        
        ### false if union not possible
        def union(n1,n2):
            p1=find(n1)
            p2=find(n2)
            if p1==p2:
                return False
            if rank[p2]>=rank[p1]:
                parent[p1]=p2
                rank[p2]+=rank[p1]
            else:
                parent[p2]=p1
                rank[p1]+=rank[p2]
            return True         ## union performed
        
        for edge in edges:
            if not union(edge[0],edge[1]):
                return False
            else:
                pass
                if self.logging: print(edge, "merge performed")
        return True
======================================================================


1061. Lexicographically Smallest Equivalent String
Didnt realize this is union find at first. And tried updating dicts at first. 
How will you realize its UNION FIND? We are creating distinct groups and each group has its own parent. 
THIS IS PRECISELY UNION FIND. Slight changes to the algo. Here we dont use rank at all as parentage is determined by lexo order and not size.

Note: find parent map also doesnt give us the ROOT parent!
Use find operation for that

ALGO_NAME: UNION_FIND
class Solution(object):
    def smallestEquivalentString(self, s1, s2, baseStr):
        list1=list(set(s1+s2))
        parent={list1[i]:list1[i] for i in range(len(list1)) }

        def find(n):
            p=parent[n]
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p = parent[p] 
            return p
            
        def union(n1,n2):
            p1=find(n1)
            p2=find(n2)
            if p1==p2:
                return False
            if p1<p2:
                parent[p2]=p1
            else:
                parent[p1]=p2
            return True
    
        for i in range(len(s1)):
            union(s1[i],s2[i])
            
        ans=""
        for i in range(len(baseStr)):
            if baseStr[i] in parent:
                ans+=find(baseStr[i])
            else:
                ans+=baseStr[i]
                
        return ans     

More questions: 
    
721. Accounts Merge

Its difficult to realize its a graph problem. What is the give away? We have to "merge" groups together. 
Union find. What are the nodes?
For every index the emails in the list are all already connected to each other. This is one unconnected component. 

For the next index the emails in that list are also connected to each other. Second unconnected component. 

Now union happens when any node in the first matches any node in the second. Now what is the parent/representative of each of the groups. first email ? index? I take index as representative/parent and merge towards bigger index. just by default it works like this. didnt change. 
Now if any elements are common these two indices will need to have a common parent after the merge operation. 

How do we find out if two pairs of indices have any common elements? We need any intersections of the lists at the two indices. Now do i need to find out n^2 pairs ? I think i dont have to by using dictionary trick. 

What if I use a hashmap to store element: index, I keep iterating and checking if each element was seen before, 
if yes I simply know at what index and I merge the current index and stored index together. 
What is the new parent? larger index here but not necessary. 
Now, I go through the dictionary again  and find index of each. for each index i find parent, but now the parentage has been updated and hence grouping can be done based on same parent value for a group. so some more dict manipulation.

class Solution(object):
    def accountsMerge(self, accounts):
        parent=[i for i in range(len(accounts))] 
        rank=[1 for i in range(len(accounts))]          ## standard rank 
        
        def find(n):
            p=parent[n]
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        
        def union(n1,n2):       ## merging to bigger index
            p1,p2=find(n1),find(n2)
            
            if p1==p2: 
                print("failed to merge") 
                return False
            
            if rank[p1]>=rank[p2]:          ##there is equal here, dont forget that
                parent[p2]=p1
                rank[p1]+=rank[p2]
            else:
                parent[p1]=p2
                rank[p2]+=rank[p1]
            print("merged",n1,n2)
            return True
        
        dict1={}
        
        for i in range(len(accounts)):
            for j in range(1,len(accounts[i]),1):
                if accounts[i][j] in dict1:
                    union(i,dict1[accounts[i][j]])
                else:
                    dict1[accounts[i][j]]=i
        dict2={}
        for key in dict1.keys():
            parent_index=find(dict1[key])
            if parent_index not in dict2:
                dict2[parent_index]=[accounts[parent_index][0]]+[key]
            else:
                dict2[parent_index].append(key)
                
        return [[x[0]]+sorted(x[1:]) for x in dict2.values()] 
======================================================================================================================
1258. Synonymous Sentences
There is no way to take all possible combinations other than to do a TREE_STYLE_DFS_ITERATIVE_BACKTRACK. Kinda got stumped on that. Tried loops lol. Otherwise same as accounts merge. 


class Solution(object):
    logging=True                
    def generateSentences(self, synonyms, text):
        ## STANDARD UNION FIND NO CHANGE
        def find(n):
            p=parent[n]
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        def union(n1,n2):
            p1,p2=find(n1),find(n2)
            if p1==p2:
                return False
            if rank[p1]>=rank[p2]:
                rank[p1]+=rank[p2]
                parent[p2]=p1
            else:
                rank[p2]+=rank[p1]
                parent[p1]=p2
            
            return True
                
        parent=[i for i in range(len(synonyms))]
        rank=[1 for i in range(len(synonyms))]
        
        dict1={}
        for i in range(len(synonyms)):
            for j in range(len(synonyms[i])):
                if synonyms[i][j] in dict1:
                    union(i,dict1[synonyms[i][j]])
                else:
                    dict1[synonyms[i][j]]=i
        
        dict2={}
        for key in dict1.keys():
            parent_index=find(dict1[key])
            if parent_index not in dict2:
                dict2[parent_index]=[key]
            else:
                dict2[parent_index].append(key)
        print("dict1",dict1)
        print("dict2",dict2)
        text=text.split()
        
        def dfs(text):
            stack=[(text,0)]
            ret=[]
            while stack:
                current,index=stack.pop()
                #create neighbor
                neighbors=[]
                if current[index] in dict1:
                    for replace_word in dict2[find(dict1[current[index]])]:
                        neighbors.append((current[:index]+[replace_word]+current[index+1:],index+1))
                else:
                    neighbors.append((current,index+1))

                if self.logging: print("current",current,index)
                if self.logging: print("neighbors",neighbors)   

                ### add to dfs stack
                for neighbor in neighbors:
                    nextCurrent,nextIndex=neighbor
                    if  nextIndex<=len(text)-1:               ## continuation
                        stack.append(neighbor)          
                    else:                                     ## leaf node ans
                        ret.append(nextCurrent)
            return sorted([" ".join(x) for x in ret])         ## sorted lexicographically

        return dfs(text)
======================================================================================================================
737. Sentence Similarity II
Same old grouping at indices 

class Solution(object):
    def areSentencesSimilarTwo(self, sentence1, sentence2, similarPairs):
        if len(sentence1)!=len(sentence2):
            return False
        
        def find(n):
            p=parent[n]
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        def union(n1,n2):
            p1,p2=find(n1),find(n2)
            if p1==p2:
                return False
            if rank[p1]>=rank[p2]:
                rank[p1]+=rank[p2]
                parent[p2]=p1
            else:
                rank[p2]+=rank[p1]
                parent[p1]=p2
            
            return True
        
        parent=[i for i in range(len(similarPairs))]
        rank=[1 for i in range(len(similarPairs))]
        
        dict1={}
        for i in range(len(similarPairs)):
            for j in range(len(similarPairs[i])):
                if similarPairs[i][j] not in dict1:
                    dict1[similarPairs[i][j]]=i
                else:
                    union(i,dict1[similarPairs[i][j]])
        #print(parent)
        
        for i in range(len(sentence1)):
            if sentence1[i]== sentence2[i]:
                continue
            if (sentence1[i] not in dict1 or sentence2[i] not in dict1):
                return False
            if find(dict1[sentence1[i]]) != find(dict1[sentence2[i]]):
                return False
        return True      
======================================================================================================================
959. Regions Cut By Slashes

Difficut question due to the conversion between numbers and grid. messy.

1. 1st task is to realize this is union find because I want to "merge things together" and count the number of unconnected components by taking count on unique parents of "updated parents list" after union find. 

2. Now the messy part is counting and converting i,j to areas in the diagram. 
For this I remove 0 based index calculate range of numbers. We have buckets here of 4 numbers and we know the number of elements in a row. So we divide by row length(4*m) to get column and divide by bucket length(4) to get column. I create the range of 4 using this. Now i get these numbers inside this buckets and merge them according to the case.
I also have to merge to the left and bottom (if it exists) at every iteration.




class Solution(object):    
    def regionsBySlashes(self, grid):
        m=len(grid)
        parent=[i for i in range(4*(m**2))] ### DONT CHANGE THIS TO 1 BASED START, YOU WILL FACE A LOT OF PAIN
        rank=[1 for i in range(4*(m**2))]
        #print(parent)
        
        def find(n):
            p=parent[n]
            # print(p)
            # print(parent[p])
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        def union(n1,n2):
            p1,p2=find(n1),find(n2)
            if p1==p2:
                return False
            if rank[p1]>=rank[p2]:
                rank[p1]+=rank[p2]
                parent[p2]=p1
            else:
                rank[p2]+=rank[p1]
                parent[p1]=p2

            return True
        def convert(i,j,m):     ### CRUX OF THE PROBLEM 
            return [i*4*m+j*4,i*4*m+j*4+1,i*4*m+j*4+2,i*4*m+j*4+3]
    
        for i in range(len(grid)):
            j=0
            while j<=len(grid[i])-1:
                print("grid[i][j]",grid[i][j])
                if grid[i][j]==" ":
                    union(convert(i,j,m)[0],convert(i,j,m)[1])
                    union(convert(i,j,m)[1],convert(i,j,m)[2])
                    union(convert(i,j,m)[2],convert(i,j,m)[3])
                elif grid[i][j]=="/":
                    union(convert(i,j,m)[0],convert(i,j,m)[3])
                    union(convert(i,j,m)[1],convert(i,j,m)[2])
                elif grid[i][j]=="\\":      ### you cant single slash here as it escapes
                    union(convert(i,j,m)[0],convert(i,j,m)[1])
                    union(convert(i,j,m)[3],convert(i,j,m)[2])
                    #j+=1
                if i+1<=m-1:
                    # print("convert(i,j,m)",convert(i,j,m))
                    # print("convert(i+1,j,m)",convert(i+1,j,m))
                    union(convert(i,j,m)[1],convert(i+1,j,m)[3])
                if j+1<=m-1:
                    union(convert(i,j,m)[2],convert(i,j+1,m)[0])
                j+=1
        
        ## after union find has been done i go see the see parent list. 
        ## I run find operation on it once so we get all root parents
        ## Then i take a unique count. 
        set1=set()
        for i in range(len(parent)):
            if find(parent[i]) not in set1:
                set1.add(find(parent[i]))
        print(set1)
        return len(set1) 

======================================================================================================================
1202. Smallest String With Swaps
Read the question properly. 


1. No Greedy solution available checked
2. Backtrack -- possible I guess? But infinite steps possible where will you terminate ?
3. This is a tricky graph problem. 


Note: The important point to note here is that if we have pairs like (a, b) and (b, c), then we can swap characters at indices a and c. Although we don't have the pair (a, c), we can still swap them by first swapping them with the character at index b


The biggest challenge in solving this problem was figuring out that, with infinite swaps, we can arrange all characters that belong to the same connected component in sorted order. With that hurdle behind us, our next challenge is, how do we find out which indices belong to the same connected component? Union Find 

1. I find the all the chars in connected component then i created a dictionary based on top parent representation. 
2. I sort this dictionary 
3. I iterate over the parent again and keep popping from the sorted list from the beginning. 


class Solution(object):
    logging = False
    def smallestStringWithSwaps(self, s, pairs):
        
        def find(n):
            p=parent[n]
            # print(p)
            # print(parent[p])
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        def union(n1,n2):
            p1,p2=find(n1),find(n2)
            if p1==p2:
                return False
            if rank[p1]>=rank[p2]:
                rank[p1]+=rank[p2]
                parent[p2]=p1
            else:
                rank[p2]+=rank[p1]
                parent[p1]=p2

            return True
        parent = [i for i in range(len(s))] #[0,1,2,3]
        rank = [1 for i in range(len(s))]    #[1,1,1,1]
        
        
        for i in range(len(pairs)):
            union(pairs[i][0],pairs[i][1])
        if self.logging: print(parent)      #[0,1,1,0]
        
        visited={}
        
        for i in range(len(parent)):
            if find(parent[i]) not in visited:
                visited[find(parent[i])]=[s[i]]
            else:
                visited[find(parent[i])].append(s[i])
        if self.logging: print(visited)      #{0: [b,d],1:[a,c]}
        
        for key in visited.keys():
            visited[key]=sorted(visited[key])
        
        ans=[None for i in range(len(s))]
        for i in range(len(parent)):
            if find(parent[i]) in visited:
                ans[i]=visited[find(parent[i])].pop(0)
        return "".join(ans)
=======================================================================================================================
990. Satisfiability of Equality Equations
1. This question requires ,me to be kinda careful. I thought how to model equality with edges. 
How to model non edges? YOU DONT NEED TO. FIRST PASS MERGE ALL THE EQUALITY. 
Second pass, check if these can be unioned if yes then return FALSE. BECUASE THESE HAVE ALREADY BEEN EQUATED. 
BUT DONT CALL UNION FOR THIS! UNION ACTUALLY STARTS changing parent even if FALSE. Check if parents are same, if YES have to return False
Again in the beginning, I dont have variable count, cant init "parent". its ok. Make parent dict and add these changes.
if n not in parent:
    parent[n]=n
    return n 
in find. Rest is same in find. 
Again in union remove rank, just compare directly. 




class Solution(object):
    logging=True
    def equationsPossible(self, equations):
        def find(n):
            if n not in parent:         """"extra as parent was empty"""
                parent[n]=n
                return n 
            p=parent[n]
            # print(p)
            # print(parent[p])
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        
        def union(n1,n2):
            p1,p2=find(n1),find(n2)
            if p1==p2:                  ##### REMOVED RANK TOTALLY
                return False
            if p1>=p2:
                parent[p2]=p1       ##dont ever put n2 here
            else:
                parent[p1]=p2

            return True
        
        parent={}
        for i in range(len(equations)):
            if equations[i][1:3]=="==":
                union(equations[i][0],equations[i][3])
        
        for i in range(len(equations)):
            if equations[i][1:3]=="!=":
                p1,p2=find(equations[i][0]),find(equations[i][3])   ##dont dare call union here to check
                if p1==p2:
                    return False
        
        return True
======================================================================
1319. Number of Operations to Make Network Connected
### graph 

### each connected component calculate the number of "extra edges" where union in UF not possible 

## For one connected graph of size, k, k-1 edges edges are needed. 
## Total number of nodes is N so we need N-1 edges total. 

1. I first thought of counting extra edges. But then I will also need size of each unconnected graph ? because
I need k1-1, k2-1, k3-1. 
2. I can simply calculate sum of k1-1,k2-1 as "Useful" graph edges
3. If i make sure that the number of cables is enough>n-1. Then it is simply. (n-1 - useful) moves

Another super smart way is to count the number of CONNECTED COMPONENTS(x). 
THEN IF WE HAVE ENOUGH CABLES, checked in the beginning, its simply x-1 cables





## Extra cables ? Yes
## less cables ? Return False
class Solution(object):
    def makeConnected(self, n, connections):
        
        if len(connections)<n-1: ##number of cables less than n-1 
            return -1
        def find(n):
            p=parent[n]
            # print(p)
            # print(parent[p])
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        def union(n1,n2):
            p1,p2=find(n1),find(n2)
            if p1==p2:
                return False
            if rank[p1]>=rank[p2]:
                rank[p1]+=rank[p2]
                parent[p2]=p1
            else:
                rank[p2]+=rank[p1]
                parent[p1]=p2

            return True
        
        parent=[i for i in range(n)]
        rank=[1 for i in range(n)]
        
        useful=0
        for i in range(len(connections)):
            if union(connections[i][0],connections[i][1])==True:
                useful+=1
        
        return  (n-1)-useful 
--------------------------------------------------------------------------------------------------------------------------------------      
        OR 
        CALCULATING NUMBER OF CONNECTED COMPONENTS
        ### REMEMBER THIS I simply need to take set of parent after doing a find operation. 
        ### or i can maintain a counter too!! decrement by 1 each time a "useful" edge is found
        
        for i in range(len(connections)):
            union(connections[i][0],connections[i][1])
        
        return len(set([find(x) for x in parent]))-1    
=======================================================================================================================

### Brute Force Approach
# Union find 
### optimised Approach 
# Union find 

### Test Case
### sorting is needed or not.

### Edge Case 
## len(logs)<n-1


Again my first thought process was to check for extra edges, return False when the first extra edge was found. 
But tracking USEFUL is more important always. Anyway extra edges can be self loops or when they do come, merging has already happened in the past. So its useless! 

Concept 2: Again you can also keep a track of "Connected components". I can maintain a counter too!! decrement by 1 each time a "useful" edge is found. When Connected components becomes 0 you found answer.



1101. The Earliest Moment When Everyone Become Friends
class Solution(object):
    def earliestAcq(self, logs, n):
        """
        :type logs: List[List[int]]
        :type n: int
        :rtype: int
        """        
        ### edge case checks
        if len(logs)<n-1:
            return -1
        
        
        logs.sort(key=lambda x: x[0])
        
        def find(n):
            p=parent[n]
            while p!=parent[p]:
                parent[p]=parent[parent[p]]
                p=parent[p]
            return p
        def union(n1,n2):
            p1=find(n1)
            p2=find(n2)
            if p1==p2:
                return False
            if rank[p2]>=rank[p1]:
                parent[p1]=p2
                rank[p2]+=rank[p1]
            else:
                parent[p2]=p1
                rank[p1]+=rank[p2]
            return True         ## union performed
        
        parent = [i for i in range(n)]
        rank = [1 for i in range(n)]
        
        useful=0
        for i in range(len(logs)):
            if union(logs[i][1],logs[i][2])==True:
                useful+=1
                if useful==n-1:
                    return logs[i][0]
        return -1
=======================================================================================================================
++++++++++++++++++++++++++++++++++++++
+++++++Group : Trie/Prefix Tree +++
++++++++++++++++++++++++++++++++++++++

This was painful. Why ? New data structures are difficult to work with. 


208. Implement Trie (Prefix Tree)

class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.endOfWord = False                      ## have to initialize as False, because we mark true later
    
class Trie(object):
    def __init__(self):
        self.root = TrieNode()                      ### initialize with empty node
        
    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char]=TrieNode()   ### create new nodes and  
            current = current.children[char]        ### moving to next node 
        current.endOfWord = True                    ### finally marking True 
                
    def search(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]       ### moving to next node 
        return current.endOfWord                   ### return True if we reach the end of word
        
    def startsWith(self, prefix):
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True
=======================================================================================================================
303. Range Sum Query - Immutable


I used an extra array, to store prefix Sums! For the edge case, i use an extra box to the left. 
This is standard in these questions. 
What is the jth column in the new array store? Sum of array starting from 0 to index j (including)


class NumArray(object):
    def __init__(self, nums):
        self.nums = [None for i in range(len(nums)+1)]
        self.nums[0]=0
        prefix=0
        for i in range(len(nums)):
            prefix += nums[i]
            self.nums[i+1] = prefix     ### i moves to i+1
            
            
    def sumRange(self, left, right):
        return self.nums[right+1]-self.nums[left]
        
Time complexity: O(1) time per query
O(n) time pre-computation.
-----------------------------------------------------------------------------------------------------------------------
Hashing all pairs can also work 
Space complexity: O(n^2)
Time complexity: O(1)
=======================================================================================================================
304. Range Sum Query 2D - Immutable


What is the ith, jth column in the new matrix store? 
Sum of rectangle  starting from 0,0 to i,j bottom corner (including)
So, m,n will have to complete matrix, 0,0 to m,n (not m-1,n-1 which is top corner)
now how to update ?





        
        
======================================================================
++++++++++++++++++++++++++++++++++++++
+++++++Group : Recursion +++
++++++++++++++++++++++++++++++++++++++


            
 Two types of Recursion

 Tail recursion (commonly used)
==========================================
 def func(n):
 	base case returns something  (returns the final ANSWER) !!!!!
 	operation             ## to create the arguments of the recursion in next line
 	retun func(n-1)

 def f(num,ans):
	if num==0:
		return ans 
	ans= num*ans 
	return f(num-1,ans)

will be called by f(5,1) 


HEAD recursion example
	def f(num):
	  ans=1
	  if num==0:   ##base case
	  	return 1
	  x=f(n-1)      ## assume we know f(n-1)
	  ans=num*x     ## opeartion comes AFTER recursion call 
	  return ans 

How to write TAIL recursion
1. First line base case. Base case will return that as if you got the answer at it. return totals, counts
2. Return of base case and the recurive function has to be the same. The answer will come from the base case
3. Processing logic: whatever operation you are doing to go to the lower recursion
4. At last line return (or sometimes do not return just call) the recursive function

How to write HEAD recursion
1. First line base case. Base case will return that as if you are starting from the end. return 0s 
2. Now we want to ASSUME that we know the answer to f(n-1) and write its answer like 
this x=f(n-1), Now x was never defined earlier so looks unintuitive but its ok
3. Write processing logic of the step from n to n-1 
4. Now opeartion using n-1 result assumed and step result calculated will give you f(n) result
5. Just return this now. 
6. HEAD recursion can save one parameter. so USE when you dont want an extra parameter. calculation of ans completes at head node. 


Head recursion
==========================================
def func(n):
	base case returns something (will return 0) or the starting point of the operation 
	x=func(n-1)                   
	operation on x 
	return x after operation  ###
======================================================================
50. Pow(x, n)
Tags: Math
------------------------------------------------------------------------------------------
Logic
x^n = (x*x)^(n/2)                         -------when n is even
x^n = x * x^(n-1)  = x * (x*x)^[(n-1)/2]  -------when n is odd
   Recursive solution.

      Base case for recursive solution, return 1 when n = 0.
------------------------------------------------------------------------------------------

Tail Recursive 
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n==0:         ## base case
            return 1
        
        if n>0:            
            return x*self.myPow(x,n-1)
        if n<0:
            return 1/(x*self.myPow(x,(-1*n)-1))
------------------------------------------------------------------------------------------
Tail Recursive Optimized
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        ## logic 
        ## syntax 
        ## edge 
        
        
        if n==0:      ### base case
            return 1
        if n<0:
            return 1/self.myPow(x,abs(n))
    
        if n%2==0:
            return self.myPow(x*x,n/2)
        else:
            return x*self.myPow(x*x,(n-1)/2)
------------------------------------------------------------------------------------------
Iterative
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        sign=0
        if n<0:
            sign=1
            n=abs(n)
        
        ans=1
        while n>0:
            if n%2==0:
                x=x*x
                n=n/2
            else:
                ans=ans*x
                x=x*x
                n=(n-1)/2

        if sign==1:
            return 1/ans
        else:
            return ans

Notice the only thing we are doing when going from recursive to iterative is updating x and n in the while loop.
But we also needed "ans" variable to keep track of the leftover x, slightly tricky. Use recursion
======================================================================
231. Power of Two

While solution 
------------------------------------------------------------------------------------------
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<=0:             ##edge cases
            return False

        
        while n>1:                ##when n=1 this loop should break and True returned
            if n%2==0:
                n=n/2
            else:
                return False
        return True
------------------------------------------------------------------------------------------       
Tail Recurse 

class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<=0:            ### edge case 
            return False 
        
        if n==1:            ### base case
            return True 
        
        if n%2!=0:
            return False
        
        return self.isPowerOfTwo(n/2)
------------------------------------------------------------------------------------------
======================================================================
326. Power of Three, 342. Power of Four
Exactly Same as Power of Two 
======================================================================
263. Ugly Number
class Solution(object):
    def isUgly(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<=0:
            return False
        while n>1:
            if n%2==0:
                n=n/2
            elif n%3==0:
                n=n/3
            elif n%5==0:
                n=n/5
            else:
                return False
        return True 
------------------------------------------------------------------------------------------        
class Solution(object):
    def isUgly(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n<=0:                    ### edge case
            return False
        
        if n==1:                    ### base case
            return True
        
        if n%2==0:                  ### base case
            n=n/2
        elif n%3==0:
            n=n/3
        elif n%5==0:
            n=n/5
        else:
            return False
        return self.isUgly(n)
        
    
====================================================================================
226. Invert Binary Tree
----------------------------------------------------------------------
We simply need to go to every node and revert left and right. so it doesnt matter which traversal you use

Simple tail recurse. Finally return root as the base stack always returns root. Rest of the stacks also return root but there returns are wasted as we dont equate to anything. 
----------------------------------------------------------------------
Done recursion, bfs/dfs later 
---------------------------------------------------------------------
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root==None:
            return 

        ## tail recurse   
        root.left,root.right=root.right,root.left
        self.invertTree(root.left)
        self.invertTree(root.right)



        ## head recurse
        a =self.invertTree(root.left)
        b= self.invertTree(root.right)
        root.left=b
        root.right=a

        
        return root

### through iteration ## dfs 
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root==None:
            return
        
        stack=[root]
        root.left,root.right=root.right,root.left
        while stack:
            current=stack.pop()
            for x in [current.left,current.right]:
                if x:
                    stack.append(x)
                    x.left,x.right=x.right,x.left
            
        return root
   
==========================================
222. Count Complete Tree Nodes
----------------------------------------------------------------------
Done head and tail recursion, rest two later 
--------------------------------------------------------------------- 
Method1: 
Head recursion 
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root==None:
            return 0
        left_count=self.countNodes(root.left)
        right_count=self.countNodes(root.right)
        
        return left_count+right_count+1
--------------------------------------------------------------------- 
Tail recursion 
class Solution(object):
    count=0                  #### you cannot take this inside 
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root==None:
            return 0
        
        self.count+=1
        self.countNodes(root.left)
        self.countNodes(root.right)
        
        return self.count   
--------------------------------------------------------------------- 
Method2:  Going from top to bottom. 
Trick is that it is given that tree is "complete"
Compare left side depth and right side depth if equal then it is a "full" tree
else recurse left and right and try finding full trees.


class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root==None:
            return 0
        
       
        countL=self.countNodes(root.left)
        countR=self.countNodes(root.right)
        
        left_depth=self.side_depth(root,True)
        right_depth=self.side_depth(root,False)
        
        ## 2^0+2^1+2^depth (1+2+4+8).. GP
        ## a (r^n-1)/r-1
        if left_depth==right_depth:
            return 2**(left_depth+1)-1
        return countL+countR+1
        
        
        
    def side_depth(self,root,left_or_not):
        count=0
        stack=[root]
        while stack:
            current=stack.pop()
            if left_or_not and current.left:
                stack.append(current.left)
                count+=1
            if current.right and not(left_or_not): 
                stack.append(current.right)
                count+=1

        return count 

Method3: More better solutons are available
I havent explored this one
class Solution:
        # @param {TreeNode} root
        # @return {integer}
        def countNodes(self, root):
            if not root:
                return 0
            leftDepth = self.getDepth(root.left)
            rightDepth = self.getDepth(root.right)
            if leftDepth == rightDepth:
                return pow(2, leftDepth) + self.countNodes(root.right)
            else:
                return pow(2, rightDepth) + self.countNodes(root.left)
    
        def getDepth(self, root):
            if not root:
                return 0
            return 1 + self.getDepth(root.left)

compare the depth between left sub tree and right sub tree.
A, If it is equal, it means the left sub tree is a full binary tree
B, It it is not , it means the right sub tree is a full binary tree
====================================================================================
Previously, we had linear stacks (recursion calls), so it was easier to define head/tail but in trees
stacks grow in a tree manner, harder to delineate
Do these questions
Binary Tree Inorder Traversal, PreOrder, PostOrder
====================================================================================    
98. Validate Binary Search Tree
Given a binary tree, determine if it is a valid binary search tree (BST).  
----------------------------------------------------------------------
Done tail recursion using global, 
---------------------------------------------------------------------
Method1: In order traversal should be striclty ascending (no duplicates allowed.)
Method2: Head recursion: If right and left are BSTs and root is between max from left side and min from right side. 
Tail is easier in this question. Why?
Tail Recursion: 
---------------------------------------------------------------------
Method1: O(N)
2 Passes 
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        inOrderTraversalList=self.isValidBSTR(root)
        
        for i in range(1,len(inOrderTraversalList)):
            if inOrderTraversalList[i]>inOrderTraversalList[i-1]:
                pass
            else:
                return False
        else:
            return True
        
    def isValidBSTR(self,root):
        if root==None:
            return []
        list1=self.isValidBSTR(root.left)
        list2=self.isValidBSTR(root.right)
        
        return list1+[root.val]+list2
---------------------------------------------------------------------
Method2: O(N), 1 pass, cheat use global or by carrying a list 

TAIL RECURSION. In recursion we either carry a list or cheat using a global as I did here

## global 
 class Solution(object):
    prev=float('-inf')
    counter=True
    def isValidBST(self, root):
        
        if root==None:
            return True
        self.isValidBST(root.left)
        if root.val<=self.prev:          
            self.counter=False
        self.prev=root.val
        self.isValidBST(root.right)
        return self.counter
---------------------------------------------------------------------
# carry a list  ## dont really try this ## overly complicated
class Solution(object):
    prev=float('-inf')

    def isValidBST(self, root):
        
        return self.isValidBSTR(root,[True])[0]
        
    def isValidBSTR(self, root,list1): 
        if root==None:
            return [True]
        self.isValidBSTR(root.left,list1)
        if root.val<=self.prev:
            list1[0]=False      ### dont do list1=[False] as this will create a new list 
        
        if list1[0]==True:      ### this is an optimization as we dont need further stacks
	        self.prev=root.val  
	        self.isValidBSTR(root.right,list1)

	    return list1
---------------------------------------------------------------------
## Iterative in order
class Solution(object):
    def isValidBST(self, root):
        prev=float('-inf')
        stack=[]
        current=root
        while current or stack:   
            if current:                # keep going left until possible
                stack.append(current)
                current=current.left
                
            else:
                current=stack.pop() 
                if current.val<=prev:
                    return False
                prev=current.val          
                current=current.right 
            
        return True
---------------------------------------------------------------------
Method2: This is not really great for this question. 

====================================================================================
100. Same Tree
----------------------------------------------------------------------
Done using head and tail recursion, Head is much cleaner
---------------------------------------------------------------------
IS THIS A HEAD RECURSION?? I am checking while going down too. So tail opeartion while going down and stopping if it is False. Head will be very expensive where we go till the end each time and then come up. Seems like a combination as there is nothing much being carried around. Using tail to check but using head to bring the solution back up. But if use a global i wont need to bring the solution back up 
---------------------------------------------------------------------
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p==None and q==None:
            return True
        if p and q==None:   ### DOnt forget these two cases 
            return False
        if q and p==None:
            return False
        
        if p.val!=q.val:
            return False     ##### THERE IS NO NEED TO CONTINUE IF THE ROOT IS NOT SAME. 
        
        
        left_ans=self.isSameTree(p.left,q.left)
        right_ans=self.isSameTree(p.right,q.right)
           
        
        
        return left_ans and right_ans

## tail recursion only and checking while going down 
class Solution(object):
    counter=True
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p==None and q==None:
            return True      #### true here is only reqd for the edge case otherwise no value of True
        if p and q==None:
            self.counter=False
        if q and p==None:
            self.counter=False
        
        if self.counter==True:
            if p.val!=q.val:
                self.counter=False

            self.isSameTree(p.left,q.left)
            self.isSameTree(p.right,q.right)
        return self.counter
           

        
====================================================================================   
 572. Subtree of Another Tree
This question isnt so easy because you need to understand that you need two functions
I need to check: left side has the subtree or right side has the subtree 
or the root itself is the tree required(same tree function reqd).


class Solution(object):
    
    def isSame(self,root1,root2):
        if not root1 and not root2:
            return True
        if (not root1 and root2) or (root1 and not root2):
            return False
        a=self.isSame(root1.left,root2.left)
        b=self.isSame(root1.right,root2.right)
        self_bool=(root1.val==root2.val)
        return a and b and self_bool
        
    def isSubtree(self, root, subRoot):
        if not root and not subRoot:
            return True
        if (not root and subRoot) or (root and not subRoot):
            return False        
        
        a=self.isSubtree(root.left,subRoot)
        b=self.isSubtree(root.right,subRoot)
        self_bool=self.isSame(root,subRoot)
        return a or b or self_bool
Time compleixty is O(S*T)
S is the number of nodes in main tree 
T is the number of nodes in subTree
====================================================================================   

HEAD RECURSION QUESTIONS
110. Balanced Binary Tree
----------------------------------------------------------------------
Done, Tricky question because just checking balanced on both sides is not enough, height has to be checked  
--------------------------------------------------------------------- 
====================================================================================
Tags: Recursion, We simply need to check at each node that left and right are balanced and the left height and right height is comparable. We want to do this from the bottom up because then height calculation has no repetition. 
---------------------------------------------------------------------

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """   
        return self.isBalancedR(root)[1]
    
    def isBalancedR(self, root):
        if root==None:
            return (0,True)         ##### base case returns 0 like the rule in head recursion 
        
        left_depth,left_balanced = self.isBalancedR(root.left)            ### we take the output from base case
        right_depth,right_balanced = self.isBalancedR(root.right)

        total_depth= max(left_depth,right_depth)+1
        total_balance= left_balanced and right_balanced and abs(left_depth-right_depth)<=1  ### we process it 
         
        return (total_depth,total_balance)     ### Then we return it 

########### CLASSSIC HEAD RECURSION 
====================================================================================
257. Binary Tree Paths
Tags: Recursion (Tail)
Given a binary tree, return all root-to-leaf paths.
Note: A leaf is a node with no children.
THIS IS A KNOWN ACROBATIC IN ANY GRAPH
---------------------------------------------------------------------
Decisions 
1. TAIL RECURSION -- HOW TO THINK ABOUT THIS
2. ANSWER IS FOUND AT EACH TAIL. SO BASE CASE RETURNS THE ANSWER AND ENDS RECURSION
3. IF I CARRY A LIST, THAT LIST WILL KEEP ADDING STUFF AT EACH RIGHT LEFT AND ADAPT ACCORDING TO THE LEVEL. THIS IS WHAT I NEED. LIST1
BUT THE ISSUE IS THAT ONCE WE GET BACK TO ROOT NODE THIS LIST IS EMPTY AGAIN. NATURE OF LIST
3. IF YOU HAVE A GLOBAL LIST, you can simply append to list at TAILS/base case then return global list as answer
4. YOU CAN ALSO CARRY THIS LIST, AND "APPEND" TO IT AT THE LAST. 
APPENDING IS DIFFERENT AND WILL CHANGE THE LIST GLOBALLY !!!!!
WHILE PASSING THE LIST WITHIN THE FUNCTION IS DIFFERENT AND MAINTAINS DIFFERENT LISTS AT DIFFERENT LEVELS!!!!
---------------------------------------------------------------------
## TWO LISTS 
class Solution(object):
    def binaryTreePathsR(self, root,list1,list2):
        if not root:
            return 
        if not root.left and not root.right:
            list1.append(str(root.val))
            list2.append("->".join(list1))
            return 
        #list1=list1+[str(root.val)]                    ### WRONG
        #self.binaryTreePathsR(root.left,list1,list2)   ### WRONG
        self.binaryTreePathsR(root.left,list1+[str(root.val)],list2)
        self.binaryTreePathsR(root.right,list1+[str(root.val)],list2)
        

    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        list2=[]
        list1=[]
        self.binaryTreePathsR(root,list1,list2)
        return list2
---------------------------------------------------------------------
## ONE GLOBAL ## ONE LIST
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        self.ret=[]   ### THIS SORT OF BEHAVES LIKE GLOBAL 
        self.binaryTreePathsR(root,[])
        return self.ret
    
    def binaryTreePathsR(self,root,pathlist):
        if root==None:
            return 
        if root.left==None and root.right==None:
            self.ret.append('->'.join(str(x) for x in pathlist+[root.val]))
        
        self.binaryTreePathsR(root.left,pathlist+[root.val])
        self.binaryTreePathsR(root.right,pathlist+[root.val])
---------------------------------------------------------------------
HEAD RECURSION GIVES VERY SIMPLE SOLUTION WITHOUT GLOBALS OR CARRYING ANYTHING!!!!
WILL HEAD RECURSE GIVE ANSWER? IF I GET ANSWER FROM LEFT AND RIGHT SIDE, IS IT EASY TO FIND ANS AT NODE? YES. THEN HEAD 
RECURSE
---------------------------------------------------------------------
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        ## base cases are tricky to write 
        
        if not root:       ## This case just catches whenever root NULL
            return []
        if not root.left and not root.right:  ## True base case
            return [str(root.val)]
        
    
        pathList1=self.binaryTreePaths(root.left)
        pathList2=self.binaryTreePaths(root.right)
        
        finalpathList=[]
        for i in pathList1:
            finalpathList.append(str(root.val)+"->" + i)
        for i in pathList2:
            finalpathList.append(str(root.val)+"->" + i)
        return finalpathList    
        
---------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK

class Solution(object):
    def dfs(self,root):
        stack=[(root,[root.val])]
        ret=[]
        while stack:
            current,path=stack.pop()
            neighbors=[]
            if current.right:
                neighbors.append((current.right,path+[current.right.val]))
            if current.left:
                neighbors.append((current.left,path+[current.left.val]))
            
            for neighbor in neighbors:
                nextCurrent,nextPath=neighbor
                if nextCurrent:
                    stack.append(neighbor) 
            if not current.left and not current.right:
                ret.append(path)
        print(ret)
        
        return  ["->".join([str(i) for i in x]) for x in ret]    
    
    
    def binaryTreePaths(self, root):
        return self.dfs(root)
---------------------------------------------------------------------

339. Nested List Weight Sum
----------------------------------------------------------------------
Difficult question, Done all 3 
---------------------------------------------------------------------
Recursion decisions 
1. Two function or one - two because we need to carry weight 
2. Tail recursion. End condition will return the ANSWER. Of course the function also needs to be RETURNED at the end. 
3. Tail has to carry two extra arguments: weight and total, while head just 1
4. Finding an integer is not base case as you keep on thinking!!! why ? we still need to recurse on other elements
---------------------------------------------------------------------
### cheating using global ### this is the best honestly ## Here also empty list is base case
## can use extra list also and append instead of self global 
class Solution(object):
    
    def depthSumR(self,nestedList,depth):
        for i in range(len(nestedList)):      ### for loop thats why integer is base case
            if nestedList[i].getInteger()!=None:
                self.sum1+=depth*nestedList[i].getInteger()
            else:
                self.depthSumR(nestedList[i].getList(),depth+1) ### recurse not on remaining list but just on individual elements

    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        self.sum1=0
        self.depthSumR(nestedList,1)
        return self.sum1

---------------------------------------------------------------------
class Solution(object):
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        return self.depthSumR(nestedList,1,0)
################### TAIL RECURSION    ### difficult
    def depthSumR(self,nestedList,weight,total):
        if len(nestedList)==0: ## base case    ### We are basically cutting the list short thats why this is the base                                                   ### case
            return total
        
        list1=[]
        for i in range(len(nestedList)):
            if nestedList[i].isInteger():
                total+=nestedList[i].getInteger()*weight.  ## we parse out all integers
            else:
                list1=list1+nestedList[i].getList() ## all complicated ones we add to the list and try again 
                                                    ## this is different than the previous solution as we recurse on the rest 

        return self.depthSumR(list1,weight+1,total)
----------------------------------------------------------------------------------------------------------------------
###################### OR HEAD RECURSION 
     def depthSumR(self,nestedList,weight):
        
        if len(nestedList)==0: ## base case 
            return 0
        list1=[]
        sum1=0
        for i in range(len(nestedList)):
            if nestedList[i].isInteger():
                sum1+=nestedList[i].getInteger()*weight
            else:
                list1=list1+nestedList[i].getList()
                
        x= self.depthSumR(list1,weight+1)
        total=sum1+x
        
        return total
====================================================================================
94. Binary Tree Inorder Traversal
----------------------------------------------------------------------
Done head and tail recursion, using while/DFS later
--------------------------------------------------------------------- 
#cheating using global
class Solution(object):
    list1=[]
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:                    ###base case
            return
        
        self.inorderTraversal(root.left)
        self.list1.append(root.val)
        self.inorderTraversal(root.right)
        
        return self.list1

--------------------------------------------------------------------- 
#carrying a list 
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ## in order 
        ### left root right 
        ##Recursion Decision 
        ## 1. two func or one -- two . I have a list I want to carry. 
        ## Tail or Head -- Tail 
        
        return self.inorderTraversalR(root,[])
    
    def inorderTraversalR(self,root,list):
        if root==None:
            return None 
        self.inorderTraversalR(root.left,list)
        list.append(root.val)
        self.inorderTraversalR(root.right,list)
        
        return list

#### THIS IS AGAIN A TAIL RECURSION !!!!!! 

CAN I WRITE A HEAD RECURSION???? THIS SHOULD SAVE ME USING THE ADDITIONAL LIST!
HOLY FUCCKKKKK!!!! I WROTE IT. TRUE BREAKTHROUGH 
SECOND THOUGHTS: Is this a true HEAD or this case is ambigous. 
This is a true head, because final answer is generated at the top. 
while in tail recurse, we dont get ans at leaves so tail is ambi. dont think about this too much, its ok to be ambi

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        
        if root==None:         #### Write base case
            return []          #### Think what kind of NUll to return 
        
        list1 = self.inorderTraversal(root.left)         #### Just assume n-1 function works magically 
        list2 = self.inorderTraversal(root.right)
        
        ans=list1+[root.val]+list2              #### use n-1 to make n ans 
        return ans                              #### return that 

====================================================================================
230. Kth Smallest Element in a BST

## BST is sorted traversal for inorder traversal
## Do inorder, return kth element
class Solution(object):
    def kthSmallest(self, root, k):
        count=0
        current=root
        stack=[]
        while current or stack:    #while loop goes on until stack is exhausted. To initialize  we give current
            if current:                 # keep going left until possible
                stack.append(current)
                current=current.left
            else:
                current=stack.pop()    # when we cannot go left any more pop ##leaf node
                count+=1
                if count==k:
                    return current.val
                current=current.right        # go right whenever left child is None
====================================================================================
671. Second Minimum Node In a Binary Tree
1. preorder, BFS didnt solve this as expected see errors 
2. DID DFS, dont return immediately, went through the entire tree. Can be optimized by using tree structure

class Solution(object):
    def findSecondMinimumValue(self, root):
        minM=root.val
        secondMin=float("inf")
        current=root
        stack=[]
        while current or stack:    #while loop goes on until stack is exhausted. To initialize  we give current
            if current:                 # keep going left until possible
                stack.append(current)
                current=current.left
            else:
                current=stack.pop()    # when we cannot go left any more pop ##leaf node
                print(current.val)
                if current.val>minM and current.val<secondMin:
                    secondMin=current.val
                current=current.right        # go right whenever left child is None
        if secondMin==float("inf"):
            return -1
        return secondMin

====================================================================================
112. Path Sum (Same as Binary Tree paths Almost)
----------------------------------------------------------------------
Done all 3,
--------------------------------------------------------------------- 
--------------------------------------------------------------------- 
1. carry an extra list 
class Solution(object):
    def hasPathSumR(self, root, targetSum,list1):
        if not root:
            return 
        if not root.left and not root.right:  ## base case leaf node
            if  sum(list1+[root.val])==targetSum:
                return True
            else:
                return False
        return self.hasPathSumR(root.left,targetSum,list1+[root.val]) or self.hasPathSumR(root.right,targetSum,list1+[root.val])
        
    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        return self.hasPathSumR(root,targetSum,[])
--------------------------------------------------------------------- 
2. no extra list by modifying targetSum each time , THIS IS HEAD RECURSION 

class Solution(object):    
    
    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        
        if not root:
            return 
        if not root.left and not root.right:  ## base case leaf node
            if  root.val==targetSum:
                return True
            else:
                return False
            
        
        return self.hasPathSum(root.left,targetSum-root.val) or self.hasPathSum(root.right,targetSum-root.val)
--------------------------------------------------------------------- 
3. We can generate a list of all sums at a node and use that 

class Solution(object):
    def hasPathSumR(self, root):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        
        if not root:
            return []
        if not root.left and not root.right:  ## base case leaf node
            return [root.val]
            
        
        list1=self.hasPathSumR(root.left) 
        list2=self.hasPathSumR(root.right)
        
        list3=[]
        for x in list1:
            list3.append(x+root.val)
        for x in list2:
            list3.append(x+root.val)
            
        return list3
    
    def hasPathSum(self, root, targetSum):
        
        sumList=self.hasPathSumR(root)
        if targetSum in sumList:
            return True
        else:
            return False
--------------------------------------------------------------------- 
Very basic using ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK and ACROBATIC 7

class Solution(object):
    def dfs(self,root,targetSum):
        if not root:
            return False
        stack=[(root,[root.val])]
        ret=[]
        while stack:
            current,path=stack.pop()
            neighbors=[]
            if current.right:
                neighbors.append((current.right,path+[current.right.val]))
            if current.left:
                neighbors.append((current.left,path+[current.left.val]))
            
            for neighbor in neighbors:
                nextCurrent,nextPath=neighbor
                if nextCurrent:
                    stack.append(neighbor) 
            if not current.left and not current.right:      ##leaf node
                ret.append(path)
        print(ret)
        
        return  any([sum(x)==targetSum for x in ret])  
    
    def hasPathSum(self, root, targetSum):
        return self.dfs(root,targetSum)
==================================================================================== 
129. Sum Root to Leaf Numbers, 
Same as Path sum , Binary Tree path 
Head recursion was simple, did it myself 
class Solution(object):
    def sumNumbersR(self, root):
        if not root:
            return []
        if not root.left and not root.right:
            return [str(root.val)]
        
        list1=self.sumNumbersR(root.left)
        list2=self.sumNumbersR(root.right)
        
        finalList=[]
        for x in list1:
            finalList.append(str(root.val)+x)
        for x in list2:
            finalList.append(str(root.val)+x)
                
        return finalList

    
    def sumNumbers(self, root):  
        return sum([int(x) for x in self.sumNumbersR(root)])
------------------------------------------------------------
Tail recurse and global list
class Solution(object):
    def sumNumbersR(self, root,str1):
        if not root:
            return
        if not root.left and not root.right:
            self.list2.append(str1+str(root.val))
            return
        
        self.sumNumbersR(root.left,str1+str(root.val))
        self.sumNumbersR(root.right,str1+str(root.val))

    
    def sumNumbers(self, root):  
        self.list2=[]
        self.sumNumbersR(root,"")
        return sum([int(x) for x in self.list2])
------------------------------------------------------------
Very basic using ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK and ACROBATIC 7

class Solution(object):
    def dfs(self,root):
        if not root:
            return False
        stack=[(root,[root.val])]
        ret=[]
        while stack:
            current,path=stack.pop()
            neighbors=[]
            if current.right:
                neighbors.append((current.right,path+[current.right.val]))
            if current.left:
                neighbors.append((current.left,path+[current.left.val]))
            
            for neighbor in neighbors:
                nextCurrent,nextPath=neighbor
                if nextCurrent:
                    stack.append(neighbor) 
            if not current.left and not current.right:      ##leaf node
                ret.append(path)
        #print(ret)
        #print(["".join([str(x) for i in x]) for x in ret])
        return  sum([int("".join([str(i) for i in x])) for x in ret])  
    
    def sumNumbers(self, root):
        return self.dfs(root)
==================================================================================== 
113. Path Sum II
----------------------------------------------------------------------
Done both 
--------------------------------------------------------------------- 
## same thing but have to carry more 
## have to use the APPEND CONCEPT WHICH MAKES THINGS GLOBAL FOR LIST2, same as before
## CAN ALSO CREATE LIST2 as global## not shown here 

class Solution(object):
    def pathSumR(self, root,list1,list2,targetSum):
        if not root:        ##base case
            return 
        
        if not root.left and not root.right: ## base case leaf node
            if sum(list1+[root.val])==targetSum:
                list2.append(list1+[root.val])
                
            
        self.pathSumR(root.left,list1+[root.val],list2,targetSum)
        self.pathSumR(root.right,list1+[root.val],list2,targetSum)
        
    def pathSum(self, root, targetSum):
        list2=[]
        self.pathSumR(root,[],list2,targetSum)
        return list2
--------------------------------------
class Solution(object):
    def pathSumR(self, root,list1,targetSum):
        if not root:        ##base case
            return 
        
        if not root.left and not root.right: ## base case leaf node
            if sum(list1+[root.val])==targetSum:
                self.list2.append(list1+[root.val])
                
            
        self.pathSumR(root.left,list1+[root.val],targetSum)
        self.pathSumR(root.right,list1+[root.val],targetSum)
    
    def pathSum(self, root, targetSum):
        self.list2=[]
        self.pathSumR(root,[],targetSum)
        
        return self.list2
--------------------------------------------------------------------- 
Very basic using ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK 
class Solution(object):
    def dfs(self,root,targetSum):
        if not root:
            return []
        stack=[(root,[root.val])]
        ret=[]
        while stack:
            current,path=stack.pop()
            neighbors=[]
            if current.right:
                neighbors.append((current.right,path+[current.right.val]))
            if current.left:
                neighbors.append((current.left,path+[current.left.val]))
            
            for neighbor in neighbors:
                nextCurrent,nextPath=neighbor
                if nextCurrent:
                    stack.append(neighbor) 
            if not current.left and not current.right:      ##leaf node
                ret.append(path)
        #print(ret)
        #print(["".join([str(x) for i in x]) for x in ret])
        return [x for x in ret if sum(x)==targetSum] 
    
    def pathSum(self, root, targetSum):
        return self.dfs(root,targetSum)

====================================================================================   
1448. Count Good Nodes in Binary Tree
This is such a straight forward question. Why did you get confused. We simply need to add a variable maxVal which will track of path max. This happens by default in tail recursion stacks
I got confused again. Bloody tricky question. 
----------------------------------------------------------------------------
class Solution(object):
    def goodNodesR(self,root,maxVal):
        if not root:
            return 
        
        if root.val>=maxVal:
            maxVal=root.val
            self.count+=1
        self.goodNodesR(root.left,maxVal)
        self.goodNodesR(root.right,maxVal)
    
    def goodNodes(self, root):
        self.count=0
        self.goodNodesR(root,float("-inf"))
        return self.count
----------------------------------------------------------------------------
class Solution(object):
    def goodNodesR(self,root,maxVal,list1):
        if not root:
            return 
        
        if root.val>=maxVal:
            maxVal=root.val
            list1[0]+=1
        self.goodNodesR(root.left,maxVal,list1)
        self.goodNodesR(root.right,maxVal,list1)
    
    def goodNodes(self, root):
        list1=[0]
        self.goodNodesR(root,float("-inf"),list1)
        return list1[0]
----------------------------------------------------------------------------
Can also be done through dfs why? even in bfs, dfs the stacks follow parent to child which is the ask here
I thought this will lead to duplicate counting, but no DFS doesnt repeat the paths it simply branches. 

Very basic using ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK 

BFS will also work. WHY ? Here I am modifying the STATE. Whatever order of traversal you choose parent will come before 
child. Thats all which is needed

class Solution(object):
    def goodNodes(self, root):
        count=1
        stack=[[root,root.val]]
        while stack:
            current=stack.pop()
            #print(current[0].left)
            for x in [current[0].left,current[0].right]:
                if x:
                    if x.val>=current[1]:
                        count+=1
                        stack.append([x,x.val])
                    else:
                        stack.append([x,current[1]])
                         
        return count
====================================================================================


====================================================================================
437. Path Sum III
1. if you simply create all root to leaf paths and then check for subarrays with given sum, it wont work!!
Why ? common stuff in two paths will be double counted
2. Head recursion wont work thought about it
3. 

====================================================================================       
690. Employee Importance
----------------------------------------------------------------------
Done both , easy ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK
---------------------------------------------------------------------
Head recursion ofcourse. Here another task is that I need to FIND the next node. Now instead of iteration. This searching can be improved by using dict. 
---------------------------------------------------------------------
ADD TO GLOBAL TAIL RECURSE

class Solution(object):
    sum1=0
    def getImportance(self, employees, id):
        for x in employees:
            if x.id==id:
                subordinates=x.subordinates
                self.sum1+=x.importance
        
        for x in subordinates:
            self.getImportance(employees,x)
        
        return self.sum1
---------------------------------------------------------------------
TAIL RECURSE AGAIN, define different function and edit global

class Solution(object):
    def getImportanceR(self, employees, id):
        for x in employees:
            if x.id==id:
                subordinates=x.subordinates
                self.sum1+=x.importance
        
        for x in subordinates:
            self.getImportanceR(employees,x)
        
    def getImportance(self, employees, id):
        self.sum1=0
        self.getImportanceR(employees,id)
        return self.sum1
---------------------------------------------------------------------
TAIL RECURSE AGAIN, using list in place of global

class Solution(object):
    def getImportanceR(self, employees, id,list1):
        for x in employees:
            if x.id==id:
                subordinates=x.subordinates
                list1.append(x.importance)
    
        for x in subordinates:
            self.getImportanceR(employees,x,list1)
        
    def getImportance(self, employees, id):
        list1=[]
        self.getImportanceR(employees,id,list1)
        return sum(list1)
---------------------------------------------------------------------
HEAD RECURSE
class Solution(object):
    def getImportance(self, employees, id):
        for x in employees:
            if x.id==id:
                if len(x.subordinates)==0:    ### base case
                    return x.importance

                totalImportance=0
                for y in x.subordinates:
                    importance=self.getImportance(employees,y)
                    totalImportance+=importance
                totalImportance+=x.importance

        return totalImportance   
---------------------------------------------------------------------
Dict solution ON time ON space

class Solution(object):
    def getImportance(self, employees, id):
        dict1={}
        for x in employees:
            dict1[x.id]=x
        
        return self.getImportanceRec(dict1,id)
        
    def getImportanceRec(self,dict1,id):
        x=dict1[id]

        if len(x.subordinates)==0:    ### base case
            return x.importance

        totalImportance=0
        for y in x.subordinates:
            importance=self.getImportanceRec(dict1,y)
            totalImportance+=importance
        totalImportance+=x.importance

        return totalImportance
----------------------------------------------------------------------
class Solution(object):
    def dfs(self,id,employees):
        stack=[id]
        ret=[]
        sum1=0
        while stack:
            current=stack.pop()
            neighbors=self.dict1[current][1]            
            for neighbor in neighbors:
                stack.append(neighbor) 
            sum1+=self.dict1[current][0]
        return sum1
    
    def getImportance(self, employees, id):
        self.dict1={}
        for employee in employees:
            self.dict1[employee.id]=[employee.importance,employee.subordinates]
        return self.dfs(id,employees)

==================================================================================== 
938. Range Sum of BST
----------------------------------------------------------------------
Done head and tail  
----------------------------------------------------------------------------------------------
Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive
----------------------------------------------------------------------------------------------
# HEAD recursion pretty obvious 
class Solution(object):
    def rangeSumBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: int
        """
        
        if root==None:
            return 0
        
        sum1=self.rangeSumBST(root.left,L,R)
        sum2=self.rangeSumBST(root.right,L,R)
        
        if root.val<=R and root.val>=L:
            ans=sum1+sum2+root.val
        else:
            ans=sum1+sum2
        
        return ans
----------------------------------------------------------------------------------------------
### optimised for BST 
    def rangeSumBST(self, root, L, R):
        if root==None:
            return 0
        
        sum1=0
        sum2=0
        rootvalue=0
        if root.val>L:  ### we get the left side only this happens
            sum1=self.rangeSumBST(root.left,L,R)   

        if root.val<R:  
            sum2=self.rangeSumBST(root.right,L,R)
        
        if root.val<=R and root.val>=L:
            rootvalue=root.val

        return sum1+sum2+rootvalue
----------------------------------------------------------------------------------------------
## easy list addition 
class Solution(object):
    def rangeSumBSTR(self, root, low, high,list1):
        if not root:
            return 0 
        
        self.rangeSumBSTR(root.left,low,high,list1)
        if root.val>=low and root.val<=high:
            list1[-1]+=root.val
            
        self.rangeSumBSTR(root.right,low,high,list1)
    
    def rangeSumBST(self, root, low, high):
        
        list1=[0]
        self.rangeSumBSTR(root, low, high,list1)
        
        return list1[-1]
----------------------------------------------------------------------------------------------
# easy using global 
class Solution(object):
    def rangeSumBSTR(self, root, low, high):
        if not root:
            return 0 
        
        self.rangeSumBSTR(root.left,low,high)
        if root.val>=low and root.val<=high:
            self.sum1+=root.val
            
        self.rangeSumBSTR(root.right,low,high)
    
    def rangeSumBST(self, root, low, high):
        
        self.sum1=0
        self.rangeSumBSTR(root, low, high)
        
        return self.sum1
----------------------------------------------------------------------------------------------
tail recurse with optimisation 
 class Solution(object):
    def rangeSumBST(self, root, L, R):
        sum1=self.inorderR(root,L,R,0)
        return sum1
        
        
    def inorderR(self,root,L,R,sum):
        if root==None:
            return sum     ### cant return 0 in tail 
        
        if root.val>R:
            sum=self.inorderR(root.left,L,R,sum)
        if root.val<=R and root.val>=L:
            sum=self.inorderR(root.left,L,R,sum)
            sum=self.inorderR(root.right,L,R,sum)
            sum+=root.val
            
        if root.val<L:
            sum=self.inorderR(root.right,L,R,sum)
        return sum 
----------------------------------------------------------------------------------------------
ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_BACKTRACK
class Solution(object):
    def dfs(self,root,low, high):
        stack=[root]
        ret=[]
        sum1=0
        while stack:
            current=stack.pop()
            neighbors=[current.left,current.right]           
            for neighbor in neighbors:
                if neighbor:
                    stack.append(neighbor) 
            if low<=current.val<=high:
                sum1+=current.val
        return sum1

    def rangeSumBST(self, root, low, high):
        return self.dfs(root,low,high)


====================================================================================  

783. Minimum Distance Between BST Nodes
530. Minimum Absolute Difference in BST
WHENEVER YOU HEAR BST YOU SHOULD ALSO HEAR INORDER
----------------------------------------------------------------------
Done 1, 2, 3
--------------------------------------------------------------------- 

1. Way too complicated 
Realized the correct logic after a LONGG time. 
What will the be min distance: minDistance from left, from right and then use max from left side and MIN from the right side.
HEAD RECURSION 
2. Simple tail inorder recursion 
3. Simple head inorder recursion 
---------------------------------------------
#Way too complicated 
class Solution(object):
    def minDiffInBST(self, root):
        return self.minDiffInBSTR(root)[0]
        
    def minDiffInBSTR(self,root):    
        if root==None:
            return float('inf'),float('-inf'),float('inf')
        
        minDist1,maxL,minL=self.minDiffInBSTR(root.left)
        minDist2,maxR,minR=self.minDiffInBSTR(root.right)
        
        
        minDist=min(minDist1,minDist2,minR-root.val,root.val-maxL)
        maxFinal=max(maxR,root.val)
        minFinal=min(minL,root.val)
        return minDist,maxFinal,minFinal
---------------------------------------------
#Simple tail inorder recursion       
class Solution(object):   
    minDiff=float('inf') #### its a minimum has to be initialized with max , CANT INITIALIZE WITH 0
    prev=float('-inf')   ### We initialize this as -inf because we are subtracting and its a min. We dont want it to succeed in the min opeartion. 
    
    def minDiffInBST(self, root):
        if root==None:
            return None 
        
        
        self.minDiffInBST(root.left)
        self.minDiff=min(root.val-self.prev,self.minDiff)
        self.prev=root.val
        self.minDiffInBST(root.right)
        return self.minDiff
---------------------------------------------
# simple head in order recursion 
class Solution(object):   
    def minDiffInBSTR(self, root):
        if not root: ## base case
            return []
        
        list1=self.minDiffInBSTR(root.left)
        list2=self.minDiffInBSTR(root.right)
        finalList=list1+[root.val]+list2
        return finalList
    
    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        finalList=self.minDiffInBSTR(root)
        minDiff=float("inf")
        for i in range(1,len(finalList),1):
            minDiff=min(finalList[i]-finalList[i-1],minDiff)
        
        return minDiff
---------------------------------------------
ALGO_NAME: TREE_STYLE_DFS_ITERATIVE_INORDER
class Solution(object):
    def dfs(self,root):
        stack=[]
        current=root
        minm=float("inf")
        prev=float("-inf")
        while current or stack:
            if current:
                stack.append(current)
                current=current.left
            else:
                current=stack.pop()
                minm = min(minm,current.val-prev)
                prev=current.val
                current=current.right
        return minm

    def minDiffInBST(self, root):
        return self.dfs(root)



---------------------------------------------
Can optimize this by using more conditions like in above 
====================================================================================
894. All Possible Full Binary Trees
Tags: HEAD recursion, Divide and Conquer 
----------------------------------------------------------------------
Very difficult question, memorize
------------------------------------------------------------------------------------------
Logic: Question seems very difficult in the beginning and we feel like taking all possible combinations. This is key. All possible combination type stuff is easy with recursion. So we start writing head recursion. 
1. We have to return a list of roots
2. Let us assume we get the list of roots from left and right for a given split. Each item in left list can be possibly combined with the right side. So we write a double for loop after getting the split and join new nodes to the left and right side
3. After that we return list of roots 
4. We also notice while splitting that we only need to go over ODD numbers. Why? Because number of nodes in 
full binary tree is always ODD.  TRY THIS TO SEE. RIGHT SIDE ODD.LEFT SIDE ODD. SUM EVEN AND THEN ROOT. SO ODD.
FULL BINARY TREES CAN ONLY BE CREATED USING ODD NODES. SO THERE IS NO POINT OF i being EVEN. 

------------------------------------------------------------------------------------------

class Solution(object):
    def allPossibleFBT(self, N):
        """
        :type N: int
        :rtype: List[TreeNode]
        """
        if N==1:
            return [TreeNode(0)]
            
        listRoots=[]
        for i in range(1,N-1,2):                    ############# GO OVER ONLY ODD NUMBERS 
            listRootsL=self.allPossibleFBT(i)                       
            listRootsR=self.allPossibleFBT(N-i-1)
            
            for i in range(len(listRootsL)):
                for j in range(len(listRootsR)):
                    newNode=TreeNode(0)
                    newNode.left=listRootsL[i]
                    newNode.right=listRootsR[j]
                    listRoots.append(newNode)
    
        
        return listRoots
====================================================================================
236. Lowest Common Ancestor of a Binary Tree
1644. Lowest Common Ancestor of a Binary Tree II

Why is this do difficult to think?

This is head recursion. start from the bottom and wherever two out of three things meet you found the answer. Left right mid. 

class Solution(object):
    def lowestCommonAncestorR(self,root,p,q):
        if not root:
            return False
        # if not root.left and not root.right:   ## base case       This is not really needed ok if you put it too
        #     if root in [p,q]:
        #         return True
        #     else:
        #         return False

        found1=self.lowestCommonAncestorR(root.left,p,q)
        found2=self.lowestCommonAncestorR(root.right,p,q)
        
        if root in [p,q]:
            root_find=True
        else:
            root_find=False
        
        
        if sorted([found1,found2,root_find])==[False,True,True]:
            self.ans=root
        
        return found1 or found2 or root_find
        
    def lowestCommonAncestor(self, root, p, q):
        self.ans=None
        self.lowestCommonAncestorR(root,p,q)
        return self.ans
----------------------------------------------------------------------  
235. Lowest Common Ancestor of a Binary Search Tree
## BST properties make this question painfully simple
class Solution(object):            
    def lowestCommonAncestor(self, root, p, q):
        if p.val>root.val and q.val>root.val:
            return self.lowestCommonAncestor(root.right,p,q)
        elif p.val<root.val and q.val<root.val:
            return self.lowestCommonAncestor(root.left,p,q)
        else:
            return root
====================================================================================
1650. Lowest Common Ancestor of a Binary Tree III
Link list question either use dict or reset to the other end. 
class Solution(object):
    def lowestCommonAncestor(self, p, q):
        dict1={}
        
        node=p
        while node:
            if node not in dict1:
                dict1[node]=1
            node=node.parent
        
        node=q
        while node:
            if node in dict1:
                return node
            node=node.parent 
====================================================================================
Strobogrammatic Number
----------------------------------------------------------------------
done
------------------------------------------------------------------------------------------
Logic :
Simply all palindromes containing 0,8,1 but account for the special case on 69 and 96
------------------------------------------------------------------------------------------
class Solution(object):
    def isStrobogrammatic(self, num):
        ### 1. palindrome but 6 is equal to 9  
        
        num=str(num)
        for i in range(len(num)/2+1):
            if (num[i]==num[len(num)-1-i] and num[i] in ('0','8','1'))  or num[i]+num[len(num)-1-i] in ['96','69']:
                pass
            else:
                return False
        else:
            return True
------------------------------------------------------------------------------------------
#Simpler solution by me with example checking 
class Solution(object):
    def isStrobogrammatic(self, num):
        """
        :type num: str
        :rtype: bool
        """
        
        i=0
        j=len(num)-1
        
        while i<=j:
            if (num[i]=="0" and num[j]=="0") or 
               (num[i]=="1" and num[j]=="1") or 
               (num[i]=="8" and num[j]=="8") or 
               (num[i]=="6" and num[j]=="9") or 
               (num[i]=="9" and num[j]=="6")  :
                i+=1
                j-=1
            else:
                return False
        
        return True 
            
        #08 False just being in 08 isnt sufficient 
        #80
        
        #0880 True 0880
        
        #6089
        
        
        #6009 true 
        #6009
        
        #06090 true
        #06090
        
        #060900 false
        #006
====================================================================================        
247. Strobogrammatic Number II
----------------------------------------------------------------------
done, memorize
------------------------------------------------------------------------------------------
Tags: Recursion, Divide and Conquer 
------------------------------------------------------------------------------------------
Logic: You see that it seems like permuatation and combination. So immediately Head recursion should come to mind.
1. We try to generate n=3 from n=2 and see we can only insert singles in the middle to do this 
2. We try to generate n=4 from n=3 and it doesnt work that well (too many conditions) so we simply try to make it n=2 
and insert pairs. We still need to insert in middle, insert in corner also works but thats just duplicates.
no need to remove 0's too because we are inserting the middle.
3. This questions seemed so impossoble when you started but it wasnt that difficut.
4. DOnt forget the case of n=1

------------------------------------------------------------------------------------------
class Solution(object):
    pairs=["00","11","88","69","96"]
    singles=["0","1","8"]
    
    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ## Case of 0 will have to be omitted
        ## n=3
        ## [1(0)1, 111, 181,  609,619,689,....]
        ## n=4
        ## take n=2 and insert all pairs in the middle
        ## 
        
        if n==1:
            return self.singles
    
        if n==2:
            return ["11","69","88","96"]
        
        if n%2==1:
            list1=self.findStrobogrammatic(n-1) ## even 
            appendlist=self.singles
        else:
            list1=self.findStrobogrammatic(n-2) ## even
            appendlist=self.pairs
        
        finallist=[]
        for i in range(len(list1)):
            for j in range(len(appendlist)):
                str1=list1[i]
                str_new=str1[:len(str1)/2]+appendlist[j]+str1[len(str1)/2:]
                finallist.append(str_new)
        
        return finallist
------------------------------------------------------------------------------------------     
class Solution(object):
    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        singles=["0","1","8"]
        pairs=["00","11","69","88","96"]
        
        if n==1:
            return singles
        if n==2:
            return ["11","69","88","96"]
        
        list2=[]
        if n%2!=0:
            list1=self.findStrobogrammatic(n-1)
            for i in list1:
                for single in singles:
                    list2.append(i[:len(i)/2]+single+i[len(i)/2:])
        else:
            list1=self.findStrobogrammatic(n-2)
            for i in list1:
                for pair in pairs:
                    list2.append(i[:len(i)/2]+pair+i[len(i)/2:]) 
====================================================================================


          
++++++++++++++++++++++++++++++++++++++
+++++++Group : Sorting/Intervals  +++
++++++++++++++++++++++++++++++++++++++

56. Merge Intervals
1. imp case: given interval can be totally inside other two intervals.
2. Dont use for loop because we will pop and insert 
3. No need to use extra space


class Solution(object):
    def merge(self, intervals):
        intervals.sort(key=lambda x: x[0]) ## key step which we can miss ## notice the syntax
        i=0
        while i<=len(intervals)-2:
            if intervals[i][1]>=intervals[i+1][0]:
                newInterval=[intervals[i][0],max(intervals[i][1],intervals[i+1][1])] ## notice this is for the case where we completely engulf 
                intervals.pop(i)
                intervals.pop(i) ## we dont increment here becuase i+1 comes to i after pop
                intervals.insert(i,newInterval)
            else:
                i+=1    ### i+1 only moves forward in else case so we have continous merging happening at i 
        return intervals

------------------------------------------------------------------------------------------     
Double while loop and violation correction in the second loop 
class Solution(object):
    def merge(self, intervals):
        intervals.sort(key=lambda x: x[0])
        logging=False
        i=0
        while i<=len(intervals)-1:
            if logging: print("i",i,"intervals",intervals)
            while i<=len(intervals)-2 and intervals[i][1]>=intervals[i+1][0]:
                intervals[i][1]=max(intervals[i][1],intervals[i+1][1])
                if logging: print("after mod",intervals)
                intervals.pop(i+1)
                if logging: print("after pop",intervals)
            i+=1
        return intervals
==============================================================================================================
57. Insert Interval
O(nlogn) solution where I simply insert and then merge operation, same as before.
No need of extra space


    def insert(self, intervals, newInterval):
        logging=False
        intervals.append(newInterval)
        intervals.sort(key=lambda x:x[0])
        i=0
        while i<=len(intervals)-1:
            if logging: print("i",i,"intervals",intervals)
            while i<=len(intervals)-2 and intervals[i][1]>=intervals[i+1][0]:
                intervals[i][1]=max(intervals[i][1],intervals[i+1][1])
                if logging: print("after mod",intervals)
                intervals.pop(i+1)
                if logging: print("after pop",intervals)
                
            i+=1
        return intervals
------------------------------------------------------------------------------------------     
O(N) solution where I only insert where its right and perform merge after that point.
ITS ALL ABOUT THE EDGE CASES!!!!!!

class Solution(object):
    def insert(self, intervals, newInterval):
        logging=False
        ####### INSERTION START
        i=0
        while i<=len(intervals)-1:
            start,end=intervals[i]
            if start>newInterval[0]:
                intervals.insert(i,newInterval)
                #if logging: print(intervals)
                break
            i+=1
        else:
            intervals.append(newInterval)
        ####### INSERTION COMPLETE
      
        #### NOW AFTER INSERTION WE NEED TO DO TWO COLLAPSES!! ONE FROM THE PREVIOUS ELEMENTS IF IT EXISTS, AND ONE FROM ITSELF
        ### WE CAN ALSO DO COLLAPSES FROM ALL ELEMENTS STARTING FROM THE PREVIOUS ELEMENT IF IT EXISTS, LIKE IN THE PREVIOUS Q. BUT NOT NEEDED.
        j=i-1
            
        while j>=0 and j<=len(intervals)-2 and intervals[j][1]>=intervals[j+1][0]:
            if logging: print(intervals)
            intervals[j][1]=max(intervals[j][1],intervals[j+1][1])
            intervals.pop(j+1)
            if logging: print(i,intervals)
        
        j+=1
        while j>=0 and j<=len(intervals)-2 and intervals[j][1]>=intervals[j+1][0]:
            if logging: print(intervals)
            intervals[j][1]=max(intervals[j][1],intervals[j+1][1])
            intervals.pop(j+1)
            if logging: print(i,intervals)
                
        return intervals
----------------------------
STEVEN POCHMAN CLASSIC ONES !! TOO GOOD! LOOK AT THE SOLUTION DIAGRAM TO UNDERSTAND
class Solution(object):
    def insert(self, intervals, newInterval):
        s, e = newInterval[0], newInterval[1]
        left = [i for i in intervals if i[1] < s]
        right = [i for i in intervals if i[0] > e]
        if left + right != intervals:
            s = min(s, intervals[len(left)][0])
            e = max(e, intervals[~len(right)][1])
        return left + [[s, e]] + right
==============================================================================================================
1229. Meeting Scheduler
0. Ofcourse we sort everything by starts
1. I listed (draw) down all examples of overlap and compared the overlap with the duration
2. If its not meeting the requirement, we need to increase one of the pointer? which one? which ends sooner!
Did a mistake in this by thinking start comparsion is needed but corrected it after error.
# [[10,50],[60,120],[140,210]]
# [[0,15],[60,70]]

# Overlap
# s1-----------------e1
#    s2-------e2

# s1-------e1
#     s2------e2
# No overlap
# -------
#             -----
class Solution(object):
    logging=False
    def minAvailableDuration(self, slots1, slots2, duration):
        slots1.sort(key=lambda x:x[0])
        slots2.sort(key=lambda x:x[0])
        
        i=0
        j=0
        
        while i<=len(slots1)-1 and j<=len(slots2)-1:
            s1,e1=slots1[i]
            s2,e2=slots2[j]
            if e2<=e1 and s2>=s1 and e2-s2>=duration:
                return [s2,s2+duration]
            elif e2>e1 and s2>=s1 and e1-s2>=duration:
                return [s2,s2+duration]
            elif e1<=e2 and s1>=s2 and e1-s1>=duration:
                return [s1,s1+duration]
            elif e1>e2 and s1>=s2 and e2-s1>=duration:
                return [s1,s1+duration]
            elif e1<=e2:
                i+=1
            elif e2<e1:
                j+=1
        
        return []    
------------------------------------------------------------------------------------------   
Simplification : Realize that the overlap region is [max(s1,s2),min(e1,e2)] in ALL CASES!!
class Solution(object):
    logging=False
    def minAvailableDuration(self, slots1, slots2, duration):
        slots1.sort(key=lambda x:x[0])
        slots2.sort(key=lambda x:x[0])
        
        i=0
        j=0
        
        while i<=len(slots1)-1 and j<=len(slots2)-1:
            s1,e1=slots1[i]
            s2,e2=slots2[j]
            if min(e1,e2)-max(s1,s2)>=duration:
                return [max(s1,s2),max(s1,s2)+duration]
            elif e1<=e2:
                i+=1
            elif e2<e1:
                j+=1
        
        return [] 




==============================================================================================================
252. Meeting Rooms
We simply sort and compare start time of next with end time of previous.
class Solution(object):
    def canAttendMeetings(self, intervals):
        intervals.sort(key=lambda x: x[0])
        
        for i in range(1,len(intervals),1):
            if intervals[i][0]<intervals[i-1][1]:
                return False
        return True
==============================================================================================================
253. Meeting Rooms II
Why is this feel so much like patience sort?
I want to minimize the number of piles. 
Initially, there are no piles. The first card dealt forms a new pile consisting of the single card.
Each subsequent card is placed on the leftmost existing pile whose top card has a value greater than or equal to the new card's value, or to the right of all of the existing piles, thus forming a new pile.
When there are no more cards remaining to deal, the game ends. This process minimizes the number of piles

Why does this work?
Because left heap no is smaller than right heap so greedy approach of filling it up ensures bigger number can go on right one (and not create a new pile)

Here we have max heaps instead of min
Here we know that pile 2 is smaller than pile1. so if new number comes its better to place it on bigger one. so that we keep the smaller one open instead of creating a new pile


WOW I CODED IT UP USING MY OWN INSPIRATION AND NO ONE HAS THOUGHT OF THIS SOLUTION BEFORE

class Solution(object):
    def minMeetingRooms(self, intervals):
        intervals.sort(key=lambda x: x[0])
        
        heap=[]
        
        for i in range(len(intervals)):
            start,end=intervals[i]
            j=0
            while j<=len(heap)-1:       ### DONT MISS THIS!!!
                if start>=heap[j]:
                    heap[j]=end
                    break
                j+=1
            else:
                heap.append(end)
        
        return len(heap)
------------------------------------------------------------------------------------------     
https://www.youtube.com/watch?v=FdzJmTCVyJU&list=PLot-Xpze53ldVwtstag2TL4HQhAnC8ATf&index=38
they sort the start and the end. whenever end happens reduce count by 1. when start happens increase count by 1. keep a track of maxcount
==============================================================================================================
435. Non-overlapping Intervals
GREEDY SOLUTION, difficult to realize saw video soln 

DOUBLE WHILE VIOLATION CORRECTION

# 1. Brute Force -- Done
# ---------------   
#     --------
#       --------

# ------          -------------
#     ----------
# 2. Optimized -- Done
# 3. Regular Test Case Dry Run -- Done 
# 4. Edge Cases  
# 5. Syntax -- Done
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        intervals.sort(key=lambda x: x[0]) #[[1,5],[1,3],[2,3],[3,4]]
        
        i=0
        count=0
        while i<=len(intervals)-2:
            
            while i+1<=len(intervals)-1 and intervals[i][1] > intervals[i+1][0]:   ##intersection strict greater only
                                                                        ### dont forget conditional check in while
                if intervals[i][1]>=intervals[i+1][1]:
                    intervals.pop(i)
                    count+=1
                else:
                    intervals.pop(i+1)
                    count+=1
            i+=1
        return count
==============================================================================================================

539. Minimum Time Difference
----------------------------------------------------------------------------------------------
concept similar to couting sort. At the end dont forget to include the difference otherway round. 
------------------------------------------------------------------------------------------
class Solution(object):
    logging=True
    def findMinDifference(self, timePoints):
        timePoints2=[]
        for x in timePoints:
            list1=x.split(":")
            list1=[int(i) for i in list1]
            timePoints2.append(list1)
        
        timePoints2.sort(key=lambda x: (x[0],x[1]))
        print(timePoints2)
        
        i=0
        minM=float("inf")
        while i<=len(timePoints2)-2:
            time2=timePoints2[i+1][0]*60+timePoints2[i+1][1]
            time1=timePoints2[i][0]*60+timePoints2[i][1]
            delta=time2-time1
            if self.logging: print(time1,time2,delta)
            #if delta<0: continue
            minM=min(minM,delta)
            i+=1
        print(minM)
        time0=timePoints2[0][0]*60+timePoints2[0][1]
        minM=min(minM,24*60-(time2-time0))    
        return minM
==============================================================================================================
670. Maximum Swap

I tried comparing with sorted list and changing the first difference which didnt work 

class Solution(object):
    def maximumSwap(self, num):
        """
        :type num: int
        :rtype: int
        """
        list1=list(str(num))
        num_sorted=sorted([int(x) for x in list1])[::-1]
        num_sorted=[str(x) for x in num_sorted]
        
        save=None
        for i in range(len(list1)):
            if not save and list1[i]!=num_sorted[i]:
                save=num_sorted[i]
                savei=i
            elif list1[i]==save:
                list1[i],list1[savei]=list1[savei],list1[i]
                break 
        return int(''.join(list1))
        
why ?       example 19999992
            sorted  99999912
            fixed   91999992 wrong answer
----------------------------------------------------------------------------------------------
Brute force
for every number find the biggest number to the right and swap
if you dont get move to the next number 
O(N**2)
----------------------------------------------------------------------------------------------
Trick is to create for 1 to 9 the right most seen index. When we see 1, we start from 9 to 2 and swap 1 with the right most available BIGGEST number. 
The right most swap from biggest number will create GREATEST NUMBER. 
We can swap with bigger number to the right to get a bigger number. We get RIGHT MOST because that takes a number from least signficance to great significance. 

class Solution(object):
    def maximumSwap(self, num):
        dict1={}
        num=str(num)
        for i in range(len(num)): ### getting the right most index for every integer
            dict1[int(num[i])]=i
        
        for i in range(len(num)):
            for j in range(9,int(num[i]),-1):    ### important to go in reverse order
                if j in dict1 and dict1[j]>i:    ### dont forget checking part
                    j=dict1[j]
                    return int(num[:i]+num[j]+num[i+1:j]+num[i]+num[j+1:])
        return int(num)
Time complexity : O(N)  
Space complexity: O(1)
==============================================================================================================
1762. Buildings With an Ocean View
I simply iterate backwards and maintain maxHeight, if current height greater then add to answer.
there is monotonic stack solutions to this question


class Solution(object):
    def findBuildings(self, heights):
        """
        :type heights: List[int]
        :rtype: List[int]
        """
        ## we simply need to maintain a right max
        ## what is the maximum right max at every index
        ## to thr right what is the heighest
        ans=[]
        maxHeight=float("-inf")
        for i in range(len(heights)-1,-1,-1):
            if heights[i]>maxHeight:
                ans.insert(0,i)
                maxHeight=max(maxHeight,heights[i])
            
                
        return ans
==============================================================================================================
791. Custom Sort String

We use extraspace for sorting. I remember this idea from before
We create a frequency list of all chars to be ordered then go through the alphabet ordering and simply add.
Dont forget the left over stuff after you are done with the letters you got

class Solution(object):
    def customSortString(self, order, s):
        dict1={}
        for i in range(len(s)):
            if s[i] not in dict1:
                dict1[s[i]]=1 
            else: 
                dict1[s[i]]+=1
        
        ans=''
        for char in order:
            if char in dict1:
                ans+=char*dict1[char]
                del dict1[char]
        
        for key in dict1.keys():
            ans+=key*dict1[key]
        return ans
==============================================================================================================
215. Kth Largest Element in an Array
------------------------------------------------------------------------------------------   
Simple heap solution
class Solution(object):
    def findKthLargest(self, nums, k):
        heap=[]
        
        for i in range(len(nums)):          ## remember we need to add all elements to the heap and keep popping once the size is k and goes to k+1
            heapq.heappush(heap,nums[i])    ## min heap by default
            if len(heap)==k+1:
                heapq.heappop(heap) ## this removes the lowest element from the heap remember as we have a min heap by default!!

### WE KEEP POPPING AFTER K SIZE IS REACHED. HOW MANY POPS DID WE MAKE? N-K THIS REVEALS THE KTH LARGEST ELEMENT
### you can pop n-k separately also but it will give the same ans
     
        return heapq.heappop(heap)          ## now the lowest element once we have gone through all and our heap size is k is our ans!
TIME COMPLEXITY: O(nlogk) ## heap push takes log k and we do it N times 
This is an improvement from O(nlogn) in sorting
------------------------------------------------------------------------------------------   
class Solution(object):
    def partition(self,array, base, pivot):  ## this function takes the right most element and puts it in its correct 
        for exp in range(base,pivot):   ## iterate from base to pivot, exclude pivot
            if array[exp] <= array[pivot]:
                array[exp], array[base] = array[base], array[exp]  ###swap with base, so that everything to right of base is bigger
                base += 1
        array[base], array[pivot] = array[pivot], array[base]       ### this step is different than base, explorer
        return base                  ## returns the pivot

[THEORY TIME
### This function gives the kth smallest value
### We can still get smallest k elements using array[:k] THESE ELEMENTS ARENT  SORTED


    def Kthsmallest_quicksort(self,array, left, right,k):           ### REMEMBER THIS FUNCTION AS A MODIFICATION TO QUICK SORT FUNCTION
        if left < right:
            pivot = self.partition(array, left, right) 
            #kth smallest number lives on k-1 index
            if pivot>k-1:                   ### In regualar quicksort we iterate on both sides, kth smallest 
                return self.Kthsmallest_quicksort(array, left, pivot-1,k)        ## now sort in other parts
                                                ## MEMORY TRICK OPPSITE SIDE HERE
            elif pivot<k-1:
                return self.Kthsmallest_quicksort(array, pivot+1, right,k)
            else:
                return array[pivot]
        else:                   ## sometimes no sorting needed due to single element remaining
            return array[left]
    ### gives the kth smallest value
   
    def findKthLargest(self, nums, k):
        return self.Kthsmallest_quicksort(nums,0,len(nums)-1,len(nums)-k+1) ## KTH LARGEST IS N-K+1 smallest
        
THEORY TIME]
==============================================================================================================
75. Sort Colors
Multiple solutions 
1. Count and store 
2. Quicksort 
3. Three pointers 
----------------------------------------------------------------------------------------------
Method 1: counting and then adding : 2 passes
class Solution(object):
    def sortColors(self, nums):
        dict1={0:0,1:0,2:0}     ## i initialize wth zeros for edge cases
        for i in range(len(nums)):
            dict1[nums[i]]+=1
        
        for i in range(len(nums)):
            if i <=dict1[0]-1:
                nums[i]=0
            elif i>dict1[0]-1 and i<=dict1[0]+dict1[1]-1:
                nums[i]=1
            else:
                nums[i]=2
----------------------------------------------------------------------------------------------       
Method2: Implement quicksort

THEORY TIME
QUICK SORT 
############################################ QUICK SORT ALGORITHM
Goal:  move all elements smaller than pivot to left of pivot,larger to right of pivot
## base is at left most, pivot is at right most , iterate from base to pivot "searching for smaller elements"
## while iteration only two things can happen, 
## less than pivot, we want things less than pivot to left of base, so we swap exp and base and move base one up
## greater than pivot, this is already good 
## once the iteration is over base is in the correct place so swap with it. 
Now goal is achieved, right side is all greater than pivot ..left side is all smaller than pivot, 
We have sorted that index correctly.Now we recurse on both sides to sort others
# Note the element which was in pivot will go to its correct position  
    P
3,1,2   ## greater than pivot, base stuck
B   
exp

    P
3,1,2   ## less than pivot 
B 
  exp
    P
1,3,2   ## swap happened so that smaller element goes to left 
    exp
  B


class Solution(object):
    def partition(self,array, base, pivot):  ## this function takes the right most element and puts it in its correct 
        for exp in range(base,pivot):   ## iterate over the from base to pivot but excludes pivot
            if array[exp] <= array[pivot]:
                array[exp], array[base] = array[base], array[exp]  ###swap with base, so that everything to right of base is bigger
                base += 1
        array[base], array[pivot] = array[pivot], array[base]
        return base                  ## returns the pivot

    def quicksort(self,array, left, right):
        if left < right:
            pivot = self.partition(array, left, right)  ## returns the index at which correct sorting has happened
            self.quicksort(array, left, pivot-1)        ## now sort in other parts
            self.quicksort(array, pivot+1, right)
    
    def sortColors(self, nums):
        self.quicksort(nums,0,len(nums)-1)        
----------------------------------------------------------------------------------------------    
Three pointers:
    
#concept 
# keep p1 at the end of 0s 
# keep p3 at the end of 2s (of left side)
# use p2 to iterate. if you find 0 or 2 move to the appropriate side 
# else just keep moving it ahead.

    

BASE EXPLORER METHOD
p1 dentoes the right side of 000000
p3 dentoes left side of 222222

class Solution(object):
    def sortColors(self, nums):
        # edge # empty list do nothing do nothing
               # single elemnt do nothing 
               # double element 
               # triple    
               
               
               
        p1=0                           ###at beginning
        p2=0
        p3=len(nums)-1                 ###and end 
        while p2<=p3 and p1<len(nums)-1:
            if nums[p1]==0:             ## shift p1 to reach end of zeros, after the last zero 
                p1+=1
            elif nums[p3]==2:           ## shift p3 to reach end of twos
                p3-=1
            elif p2<p1:                 ###sometimes p2 will fall behind correct it   
                p2=p1
                
            elif nums[p2]==1:           ### explore more and do nothing
                p2+=1
            elif nums[p2]==0:           ### 0 found  swap with p1
                nums[p1],nums[p2]=nums[p2],nums[p1]
            elif nums[p2]==2:           ### 2 found  swap with p3
                nums[p3],nums[p2]=nums[p2],nums[p3]
==============================================================================================================

347. Top K Frequent Elements

You have to return "K most frequent" 
That means you have to return K distinct values which are most frequent. Arrange all elements according to frequency and then choose topK

i found min frequency.
Then went over all keys to find till which frequency i should add to ans.


1. Sorted the frequencies using sort
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        dict1={}
        for i in range(len(nums)):
            if nums[i] not in dict1:
                dict1[nums[i]]=1
            else:
                dict1[nums[i]]+=1
        
        freq_min=sorted(dict1.values())[::-1][k-1]      ## find the minimum allowed frequency
                                                        ## here I am simply sorting frequencies and choosing k top frequency
                                                        ## What happens if a given frequency has many values. Then there will duplicates in freq values
                                                        ## but this will still work
        ans=[]
        for key in dict1.keys():
            if dict1[key]>=freq_min:
                ans.append(key)
        return ans 
Time complexity: O(nlogn) for sorting values 
------------------------------------------------------------------------------------------    
2. Sorted the frequencies by going from top to bottom
Now instead of sorting frequency, we can simply go from top to bottom as we know the upper limit. We revert the key value in dictionary for this.
from collections import Counter
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        dict1=Counter(nums)
        dict2={}
        for key in dict1.keys():
            if dict1[key] not in dict2:
                dict2[dict1[key]]=[key]
            else:
                dict2[dict1[key]].append(key)
        arr1=[]
        for freq in range(len(nums),0,-1): ## instead of sorting freq we go from highest to lowest and break when len of array is k 
            if freq in dict2:
                for val in dict2[freq]:
                    arr1.append(val)
                    if len(arr1)==k:
                        return arr1
------------------------------------------------------------------------------------------   
3. Sorted frequencies by using heap
THEORY TIME
Heap solution since we are talking about top K using heapq.heappop(heap,(key,value)) and heapq.heappush(heap,(key,value))
THEORY TIME
We want k largest numbers. Which heap? min heap of size k 


from collections import Counter
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        dict1=Counter(nums)
        heap=[]
        
        ## WE CREATE A K MINHEAP LIKE THIS KEEP PUSHING ELEMENTS ON THE HEAP AND THEN POP WHEN THE LENGTH INCREASES TO K+1
        ## POPPING REMOVES THE SMALLEST ELEMENT
        for key in dict1: # O(N)
	        heappush(heap, (dict1[key], key)) # freq, item - O(log(k)) ##frequency goes first as that is the key on which heap is organised
	        if len(heap)==k+1:  ## This is going to pop n-k times, whats left after this ? top K
		        heappop(heap)   ## popping removes minimum element out of k+1 because min heap by default
        
        
        return [x[1] for x in heap] ### THIS IS NOT ORDERED OFCOURSE, IF YOU WANT ORDERED KEEP HEAPPPOPING WHICH WILL GIVE YOU SMALLEST TO LARGEST.
        # res = []                  ### LIKE THIS
        # while heap: # O(k)
        #     frq, item = heappop(heap) # O(logk)
        #     res.append(item)
        # return res
------------------------------------------------------------------------------------------  
4. Sorted frequency USING QUICKSELECT as a modification to the first method to find kth without sorting
i want kth largest, which is n-k+1 smallest. lets get that. this is min freq


from collections import Counter
class Solution(object):
    
    def partition(self,array, base, pivot):  ## this function takes the right most element and puts it in its correct 
        for exp in range(base,pivot):   ## iterate over the from base to pivot but excludes pivot
            if array[exp] <= array[pivot]:
                array[exp], array[base] = array[base], array[exp]  ###swap with base, so that everything to right of base is bigger
                base += 1
        array[base], array[pivot] = array[pivot], array[base]
        return base                  ## returns the pivot
    def Kthsmallest_quicksort(self,array, left, right,k):           ### REMEMBER THIS FUNCTION AS A MODIFICATION TO QUICK SORT FUNCTION
        if left < right:
            pivot = self.partition(array, left, right) 
            #kth smallest number lives on k-1 index
            if pivot>k-1:                                           ### In regualar quicksort we iterate on both sides, kth smallest 
                return self.Kthsmallest_quicksort(array, left, pivot-1,k)        ## now sort in other parts
            elif pivot<k-1:
                return self.Kthsmallest_quicksort(array, pivot+1, right,k)
            else:
                return array[pivot]
        else:                   ## sometimes no sorting needed due to single element remaining
            return array[left]
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        dict1=Counter(nums)
        array=dict1.values()
        freq_min=self.Kthsmallest_quicksort(array,0,len(array)-1,len(array)-k+1) ## i want kth largest so n-k+1
        
        ans=[]
        for key in dict1.keys():
            if dict1[key]>=freq_min:
                ans.append(key)
        return ans 
------------------------------------------------------------------------------------------  
from collections import Counter
class Solution(object):
    
    def partition(self,array, base, pivot):  ## this function takes the right most element and puts it in its correct 
        for exp in range(base,pivot):   ## iterate over the from base to pivot but excludes pivot
            if array[exp][1] <= array[pivot][1]:    ##the only change
                array[exp], array[base] = array[base], array[exp]  ###swap with base, so that everything to right of base is bigger
                base += 1
        array[base], array[pivot] = array[pivot], array[base]
        return base                  ## returns the pivot
    def Kthsmallest_quicksort(self,array, left, right,k):           ### REMEMBER THIS FUNCTION AS A MODIFICATION TO QUICK SORT FUNCTION
        if left < right:
            pivot = self.partition(array, left, right) 
            #kth smallest number lives on k-1 index
            if pivot>k-1:                                           ### In regualar quicksort we iterate on both sides, kth smallest 
                return self.Kthsmallest_quicksort(array, left, pivot-1,k)        ## now sort in other parts
            elif pivot<k-1:
                return self.Kthsmallest_quicksort(array, pivot+1, right,k)
            else:
                return array[pivot]
        else:                   ## sometimes no sorting needed due to single element remaining
            return array[left]
    def topKFrequent(self, nums, k):
        dict1=Counter(nums)
        array=[(key,dict1[key]) for key in dict1]
        self.Kthsmallest_quicksort(array,0,len(array)-1,len(array)-k)
        return [x[0] for x in array[len(array)-k:]] 
        
## how ?
# i want top k 
# i know how to get bottom k arr[:k]
# if you get bottom n-k arr[:n-k]
# whats left ? k elements ## arr[n-k:]


==============================================================================================================
973. K Closest Points to Origin
Heap THEORY
We want k minimum points. Which heap? max heap of size k . Counterintuitive?
We want k largest numbers. Which heap? min heap of size k 
Heap THEORY

class Solution(object):
    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        heap=[]
        
        for i in range(len(points)):
            point=points[i]
            sqDistance=(point[0]**2+point[1]**2)*-1 ### multplying by -1 because we need a max heap not min heap
            heapq.heappush(heap,(sqDistance,point))
            if len(heap)==k+1:                      ## pop n-k times the maximum 
                heapq.heappop(heap)                 ### we pop the maximum out 
        return [x[1] for x in heap]                 ### we are left with the minimum k 
------------------------------------------------------------------------------------------  
Quickselect
1. I used list of tuples
## Note there can be duplicate points 
class Solution(object):
    def partition(self,array, base, pivot):  ## this function takes the right most element and puts it in its correct 
        for exp in range(base,pivot):   ## iterate over the from base to pivot but excludes pivot
            if array[exp] <= array[pivot]:
                array[exp], array[base] = array[base], array[exp]  ###swap with base, so that everything to right of base is bigger
                base += 1
        array[base], array[pivot] = array[pivot], array[base]
        return base                  ## returns the pivot
    def Kthsmallest_quicksort(self,array, left, right,k): ### REMEMBER THIS FUNCTION AS A MODIFICATION TO QUICK SORT FUNCTION
        if left < right:
            pivot = self.partition(array, left, right) 
            #kth smallest number lives on k-1 index
            if pivot>k-1:                                           ### In regualar quicksort we iterate on both sides, kth smallest 
                return self.Kthsmallest_quicksort(array, left, pivot-1,k)        ## now sort in other parts
            elif pivot<k-1:
                return self.Kthsmallest_quicksort(array, pivot+1, right,k)
            else:
                return array[pivot]
        else:                   ## sometimes no sorting needed due to single element remaining
            return array[left]
    def kClosest(self, points, k):
        array=[]
        
        for i in range(len(points)):
            point=tuple(points[i])
            #print(point)
            sqDistance=(point[0]**2+point[1]**2)
            array.append((point,sqDistance))
            
        sqDistanceList=[x[1] for x in array]
        dist_max= self.Kthsmallest_quicksort(sqDistanceList,0,len(sqDistanceList)-1,k)
        
        for i in range(len(array)-1,-1,-1): ## always go backwards while popping i like this 
            if array[i][1]>dist_max:
                array.pop(i)
        return [x[0] for x in array]
------------------------------------------------------------------------------------------                 
I use a dictionary instead of list of tuples. this is so much more messy implementation wise.
But is faster. Why? I am guessing there are many duplicates in test cases which are faster in dict check

class Solution(object):
    def partition(self,array, base, pivot):  ## this function takes the right most element and puts it in its correct 
        for exp in range(base,pivot):   ## iterate over the from base to pivot but excludes pivot
            if array[exp] <= array[pivot]:
                array[exp], array[base] = array[base], array[exp]  ###swap with base, so that everything to right of base is bigger
                base += 1
        array[base], array[pivot] = array[pivot], array[base]
        return base                  ## returns the pivot
    def Kthsmallest_quicksort(self,array, left, right,k): ### REMEMBER THIS FUNCTION AS A MODIFICATION TO QUICK SORT FUNCTION
        if left < right:
            pivot = self.partition(array, left, right) 
            #kth smallest number lives on k-1 index
            if pivot>k-1:                                           ### In regualar quicksort we iterate on both sides, kth smallest 
                return self.Kthsmallest_quicksort(array, left, pivot-1,k)        ## now sort in other parts
            elif pivot<k-1:
                return self.Kthsmallest_quicksort(array, pivot+1, right,k)
            else:
                return array[pivot]
        else:                   ## sometimes no sorting needed due to single element remaining
            return array[left]
    def kClosest(self, points, k):
        dict1={}
        
        for i in range(len(points)):
            point=tuple(points[i])
            #print(point)
            sqDistance=(point[0]**2+point[1]**2)
            if point not in dict1:
                dict1[point]=[sqDistance,1]
            else:
                dict1[point][1]+=1
                
                
        array=[]     
        for key in dict1:
            for j in range(dict1[key][1]):
                array.append(dict1[key][0])
        
        dist_max= self.Kthsmallest_quicksort(array,0,len(array)-1,k)
        
        ans=[]
        for point in dict1.keys():
            if dict1[point][0]<=dist_max:
                for j in range(dict1[point][1]):
                    ans.append(list(point))
        return ans 
------------------------------------------------------------------------------------------ 
Another way is to modify the partition algo slightly
## Note there can be duplicate points 
class Solution(object):
    def partition(self,array, base, pivot):  
        for exp in range(base,pivot):   
            if array[exp][1] <= array[pivot][1]: #### CHANGE HERE##########################
                array[exp], array[base] = array[base], array[exp]  
                base += 1
        array[base], array[pivot] = array[pivot], array[base]
        return base                  ## returns the pivot
    
    def Kthsmallest_quicksort(self,array, left, right,k): ### REMEMBER THIS FUNCTION AS A MODIFICATION TO QUICK SORT FUNCTION
        if left < right:
            pivot = self.partition(array, left, right) 
            #kth smallest number lives on k-1 index
            if pivot>k-1:                                           ### In regualar quicksort we iterate on both sides, kth smallest 
                return self.Kthsmallest_quicksort(array, left, pivot-1,k)        ## now sort in other parts
            elif pivot<k-1:
                return self.Kthsmallest_quicksort(array, pivot+1, right,k)
            else:
                return array[pivot]
        else:                   ## sometimes no sorting needed due to single element remaining
            return array[left]
        
    def kClosest(self, points, k):
        array=[]
        
        for i in range(len(points)):
            point=tuple(points[i])
            #print(point)
            sqDistance=(point[0]**2+point[1]**2)
            array.append((point,sqDistance))
        self.Kthsmallest_quicksort(array,0,len(array)-1,k)
        return [x[0] for x in array[:k]]    ### top K elements, we can easily get using this
==============================================================================================================
414. Third Maximum Number
def thirdMax(self, nums):
        nums=list(set(nums))  ## asking for distinct maximum so remove dups
        k=len(nums)-3+1 ##
        if k>=1:        ## k has to be valid
            return self.Kthsmallest_quicksort(nums,0,len(nums)-1,k)
        else:
            return max(nums)
==============================================================================================================

                
++++++++++++++++++++++++++++++++++++++
+++++++Group : LINK LIST +++
++++++++++++++++++++++++++++++++++++++      
        

876. Middle of the Linked List
classic mid point finder technique. Used in other questions. 
Done
==============================================================================================================
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow=head
        fast=head
        while fast and fast.next:
            slow=slow.next 
            fast=fast.next.next
        
        return slow 
==============================================================================================================        

        
237. Delete Node in a Linked List
----------------------------------------------------------------------------------------------------------------
head is not given. what trickery is this? Instead of deleting this node delete the next one. but copy the value of the next one on this one before the delete 
--------------------------------------------------------------------------------------------------------------------    
 class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val=node.next.val
        node.next=node.next.next
==============================================================================================================
203. Remove Linked List Elements
## we cant use the above trickery here 
## this is a really really annoying question despite being and looking so easy

DOUBLE WHILE VIOLATION CORRECTION WORKS BEST WHEREVER YOU DONT FORWARD UNTIL A CORRECTION IS DONE

class Solution(object):
    def removeElements(self, head, val):
        
        dummy=ListNode(0)
        dummy.next=head
        node=dummy
        while node:
            while node.next and node.next.val==val:
                node.next=node.next.next
            node=node.next
        return dummy.next

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        a=ListNode(float('inf')) ### this is needed because we might remove head also 
        a.next=head
        
        prev=a
        node=head
        while node:
            if node.val==val:
                prev.next=node.next
            else:
                prev=node  ### only update prev in this case not always      ### tricky as i always update       
            node=node.next
            
        return a.next

----------------------------------------------------------------------------------------------------------------

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        a=ListNode(float('inf'))
        a.next=head
        
        
        node=a
        while node and node.next:   ##### we need to check for node.next as we are using .next.next below
            if node.next.val==val:
                node.next=node.next.next
            else:     
                node=node.next #### we dont move ahead always as the next number also might be neeed to remove 
            
        return a.next

 ## since we are using node.next loop will terminate at the last node. but that is okay as we have already checked till node. node.next is where check happening 
==============================================================================================================
83. Remove Duplicates from Sorted List similar to Remove Linked List Elements
it is sorted so we only need to remove next nodes 

VERY CLEAN DOUBLE WHILE VIOLATAION CORRECTION

class Solution(object):
    def deleteDuplicates(self, head):
        node=head
        while node:
            while node.next and node.next.val==node.val:
                node.next=node.next.next
            node=node.next
        return head




class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        node=head
        prev=ListNode(float("inf")
        while node:
            if node.val==prev:
                prev.next=node.next
            else:
                prev=node
            node=node.next

----------------------------------------------------------------------------------------------------------------

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
       
        
        node=head
        while node and node.next:  ##### we need to check for node.next as we are using .next.next below ## check for node is needed for edge case 
            if node.next.val==node.val:
                node.next=node.next.next
            else:
                node=node.next #### we dont move ahead always as the next number also might be neeed to remove 
            
            
        return head
==============================================================================================================    
141. Linked List Cycle

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        dict1={}
        node=head
        while node:
            if node not in dict1:
                dict1[node]=1
            else:
                return True
            node=node.next
        return False
----------------------------------------------------------------------------------------------------------------

TWO pointer SOLUCHAN -- DUN DUN DUN DUN
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head==None:
            return False
        
        slow=head
        fast=head 
        while fast and fast.next: ### have to check for fast before calling fast.next 
            slow=slow.next
            fast=fast.next.next  ### have to check for fast.next above
            if slow==fast:
                return True
        return False
==============================================================================================================

class Solution(object):
    def removeNthFromEnd(self, head, n):
        dummy=ListNode(0)
        dummy.next=head
        
        
        count=0
        fast=dummy
        while fast:
            if count==n+1:
                break
            count+=1
            fast=fast.next
        slow=dummy
        while fast:
            slow=slow.next
            fast=fast.next
        
        slow.next=slow.next.next
        return dummy.next
==============================================================================================================
206. Reverse Linked List
This is a very simple question which bothered me a lot as I tried to switch using tuples but still the order mattered. 
What is one operation in Reverse Link list:
P     N
NULL  1->2->3->4->5->NULL
      P  N
NULL<-1  2->3->4->5
This diagram is crucial remember this

WE HAVE 3 UPDATES. JUST BE CAREFUL WHILE UPDATING
----------------------------------------------------------------------------------------------------------------
class Solution(object):
    def reverseList(self, head):
        
        prev=None        
        node=head
        while node:
            tmp=node.next   ###order is important
            node.next=prev
            prev=node
            node=tmp

            # or node.next,prev,node=prev,node,node.next  ### IF you change the order it wont work. Idk why 
            ### something to do with pass by reference i guess  
    
            
        return prev  ### whenever you write code with while node. Node is null at the end
----------------------------------------------------------------------------------------------------------------
Recursion: Follow head recursion logic to make this 

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None:           ###  edge case
            return 
        
        if head.next==None:      ##### base case is not head NULL  
            return head
        
        root1=self.reverseList(head.next)
        
        tmp=head.next
        head.next=None
        tmp.next=head
        
        
        return root1

==============================================================================================================
234. Palindrome Linked List
Done 
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        
        ## Find middle node 
        slow, fast=head, head
        while fast and fast.next:
            slow=slow.next
            fast=fast.next.next
        
        ## iterate again 
        left=head 
        right=slow
        list1=[]
        list2=[]
        while right:
            list1.append(left.val)
            list2.append(right.val)
            left=left.next
            right=right.next
        
        return list1==list2[::-1]
----------------------------------------------------------------------------------------------------------------

#1. Find the middle node (Using standard fast pointer, slow pointer technique)
#2. reverse the second half list (again standard reversal) 
#3. Compare the two halves one by one 

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        ## edge cases-- one node, no node  will return True as expected 


        ### standard way of finding mid point 
        slow=head
        fast=head
        while fast and fast.next:  ### standard twice moving technique 
            slow=slow.next
            fast=fast.next.next
        
        ### Standard Reverse link list always use this code

        prev=None
        node=slow
        while node:
            tmp=node.next
            node.next=prev
            prev=node
            node=tmp
        
        ### Check for equality 
        ptr1=head
        ptr2=prev   ### I use prev and not node as as end of the reversal node is NULL (also obvious from while node)
        while ptr1 and ptr2:
            if ptr1.val!=ptr2.val:
                return False
            ptr1=ptr1.next
            ptr2=ptr2.next
        
        
        return True
==============================================================================================================    
21. Merge Two Sorted Lists
Difficult 
1. Do it in place is possible dont try to copy 
2. Initialize an  extra head (because you dont know which is the first) and start operations on the next 
Similar to merge trees question 

3. I find list pointer management difficult!!!!! why ?? which subroutine is common
4. You need an extra point to keep track of head of the list and add the minimum to this pointer.
This is different than merge array 

  
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        
        ptr1=l1
        ptr2=l2
        ptr3=ListNode(0)
        a=ptr3
        while ptr1 and ptr2:
            if ptr1.val<=ptr2.val:
                ptr3.next=ptr1
                ptr1=ptr1.next
            else:
                ptr3.next=ptr2
                ptr2=ptr2.next
            
            ptr3=ptr3.next
        
        if ptr2:
            ptr3.next=ptr2
        
        if ptr1:
            ptr3.next=ptr1
        
        
        return a.next

Group: merge lists
==============================================================================================================     
160. Intersection of Two Linked Lists
Done both ways
----------------------------------------------------------------------------------------------------------------
Using Dict: O(m+n) Time, max(O(m),O(n)) space

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        dict1={}
        
        node=headA
        while node:
            if node not in dict1:
                dict1[node]=1
            node=node.next
            
        node=headB
        while node:
            if node in dict1:     ### dont try to add it again to the dict 
                return node
            node=node.next
            
        return None
        
        
----------------------------------------------------------------------------------------------------------------     
#### Super Duper Clever approach 
by seeing that the total length of the loop is same. shift ptrA to headB when it reaches None and vice versa.
#### Either they can meet together or they will both end up as None simulataneously. Only two possibilities!!!!!!

I missed the end up as None simulataneously

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
    
        
        ptr1=headA
        ptr2=headB
        while True:
            if ptr1 and ptr2:
                if ptr1==ptr2:
                    return ptr1
                ptr1=ptr1.next
                ptr2=ptr2.next
            elif ptr1 and not ptr2:
                ptr2=headA
            elif ptr2 and not ptr1:
                ptr1=headB
            else:
                return None  ### both can reach none at same time if they are separate lists and end here ## easy to miss  
==============================================================================================================
 
2. Add Two Numbers
Creating new nodes everywhere 
### extra code can be removed by considering zeros where one number is smaller 


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        ptr1=l1
        ptr2=l2
        carry=0
        prev=ListNode(0)
        a=prev
        
        while ptr1 and ptr2:
            newNode=ListNode(0)
            sum1=ptr1.val+ptr2.val+carry 
            newNode.val=sum1%10
            carry=sum1/10
            prev.next=newNode 
            ptr1=ptr1.next
            ptr2=ptr2.next
            prev=newNode
            
        while ptr2==None and ptr1:
            newNode=ListNode(0)
            sum1=ptr1.val+carry 
            newNode.val=sum1%10
            carry=sum1/10
            prev.next=newNode
            ptr1=ptr1.next
            prev=newNode
        
        while ptr1==None and ptr2:
            newNode=ListNode(0)
            sum1=ptr2.val+carry 
            newNode.val=sum1%10
            carry=sum1/10
            prev.next=newNode
            ptr2=ptr2.next
            prev=newNode

        
        if ptr1==None and ptr2==None and carry==1:
            newNode=ListNode(carry)
            prev.next=newNode
        
        return a.next   

### cheat soluchan by converting to strings 

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        #no neg assumption 
        
        str1=""
        
        node=l1
        while node:
            str1+=str(node.val)
            node=node.next 
        
        str2=""
        node=l2
        while node:
            str2+=str(node.val)
            node=node.next 
        
        final_str=str(int(str1[::-1])+int(str2[::-1]))[::-1]
        
        prev=ListNode(0)
        a=prev
        
        for i in range(len(final_str)):
            newNode=ListNode(0)
            newNode.val=int(final_str[i])
            prev.next=newNode
            prev=newNode
            
        return a.next 
==============================================================================================================
430. Flatten a Multilevel Doubly Linked List
simple dfs 

class Solution(object):
    def flatten(self, head):
        if not head:
            return 
        dummy=Node(0)
        prev=dummy
        
        stack=[head]
        while stack:
            current=stack.pop()
            for x in [current.next,current.child]:    ##LIFO   ### This is like preorder [right,left]
                if x:
                    stack.append(x)
            prev.next=current
            current.prev=prev
            current.child=None
            prev=current
        dummy.next.prev=None #### THIS PART IS CRUCIAL as the first element 
        
        return dummy.next 
        
--------------------------------------------------------
create an extra list
class Solution(object):
    def flatten(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head: ##edge case
            return 
        
        list1=[]
        stack=[head]
        while stack:
            current=stack.pop()
            for x in [current.next,current.child]:    ##LIFO 
                if x:
                    stack.append(x)
            list1.append(current.val)
        
        prev=Node(list1[0])
        prev.child=None
        a=prev
        
        for i in range(1,len(list1),1):
            newNode=Node(list1[i])
            prev.next=newNode
            newNode.prev=prev
            newNode.child=None
            prev=newNode
        return a
        
==============================================================================================================    
138. Copy List with Random Pointer
Logic : 
Question is exactly same as Clone graph. 
When we copy during iteration. pointers keep pointing to the old nodes. We need to update this. 

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if head==None:   ### Edge case 

            return 


        dict1={}
        
        a=ListNode(0)
        a.next=head
        
        node=head
        while node:
            newNode=ListNode(0)
            newNode.val=node.val
            newNode.next=node.next
            newNode.random=node.random
            dict1[node]=newNode            
            node=node.next
        
        list1=dict1.values() ##### values not keys
        for i in range(len(list1)):
            if list1[i].next:           ###### This is NEEDED becuase it is possible at the end case
                list1[i].next=dict1[list1[i].next]
            if list1[i].random:   
                list1[i].random=dict1[list1[i].random]
        
        return dict1[a.next]
----------------------------------------------------------------------------------------------------------------
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        
        dict1={}
        
        node=head
        dummy=Node(0)
        prev=dummy
        while node:
            newNode=Node(node.val,random=node.random)
            dict1[node]=newNode
            prev.next=newNode
            prev=newNode
            node=node.next
            
        node=dummy.next
        while node:
            if node.random in dict1:                    ### i missed this part ## when random node points to None dict cant be updated with it 
                node.random=dict1[node.random]
            else:
                node.random=None
            node=node.next
        return dummy.next

==============================================================================================================     
24. Swap Nodes in Pairs
----------------------------------------------------------------------------------------------------------------
Essentially you have to make swaps. Create logic of swapping first
a->1->2->3>4->5->6->7->8


P N---->
a 1<-2 3->4->5->6->7->8   (3 swaps)
----->

      P  N
a->2->1->3->4->5->6->7->8 (after swap positions)

what happens if odd number of nodes. Last node wont be inverted. but logic will still work 

----------------------------------------------------------------------------------------------------------------
MAGICAL SOLUTION I FOUND!!! I JUST STAND BACK AND RECORD ALL POINTERS, SAVE TO VARIABLES 
AND THEN UPDATE IT BACK ACCORDING TO THE NEW POSITIONS!! TOO EASY

class Solution(object):
    def swapPairs(self, head):
        dummy=ListNode(0)
        dummy.next=head
        
        node=dummy
        while node and node.next and node.next.next:
            a=node.next #1              ### LOL SITTING AT DUMMY I RECORD EVERYTHING 
            b=node.next.next #2
            c=node.next.next.next #3
            #print(a.val,b.val,c.val)
                                        ##LOL SITTING AT DUMMY I JUST UPDATED NEW POSITIONS
            node.next=b #2
            node.next.next=a ##1
            node.next.next.next=c ##3
            
            node=a                      ### UPDATE DUMMY TO ONE LOCATION BEFORE SWAP
        return dummy.next

==============================================================================================================
92. Reverse Linked List II
done
----------------------------------------------------------------------------------------------------------------
Logic : we just need to reverse between two points and ajust the end nodes after that 

#       3     5 
# #  P  N
# a->1->2->3->4->5->6->7->8

# a->1->2->
Note in reversal: when we go from node to next. we change the node.next pointer 

----------------------------------------------------------------------------------------------------------------
class Solution(object):
    def reverseBetween(self, head, m, n):
        a=ListNode(0) ### why is dummy needed ? because it is possible to revert from the first node 
        a.next=head
        prev=a         
        
        count=0
        node=head 
        
        while node:
            count+=1
            if count>=m and count<=n:  ## both equalities because of the note above
                if count==m:
                    prev1=prev     ## I record the prev1 node. dont need to record the start1 node as it is connected to prev1 
                if count==n: 
                    end1=node      ### I record the end node 
                    after1=node.next ### I record the after node also as it will be lost 
                if count==n+1:     ##### No need to traverse after you have inverted till the NTh point 
                	break      
                    
                temp=node.next    #### classic link list reversal 
                node.next=prev
                prev=node
                node=temp
            else:
                prev=node
                node=node.next 
        
        prev1.next.next=after1      ### adjustments after this 
        prev1.next=end1
    
        return a.next        ## classic dummy next return 
        
        
-----------------------------------------
class Solution(object):
    def reverseBetween(self, head, left, right):
        dummy=ListNode(0)
        dummy.next=head
        
        
        node1=dummy
        count1=0
        while node1:
            count1+=1
            if count1==left:
                break
            node1=node1.next
            
    
        node2=node1.next
        prev=node1
        while node2:
            count1+=1
            tmp=node2.next
            node2.next=prev
            prev=node2
            node2=tmp
            if count1==right+1:
                break
        
        tmp=node1.next
        node1.next=prev
        tmp.next=node2
        
        return dummy.next
==============================================================================================================  
708. Insert into a Sorted Circular Linked List
Seemingly simple question 
3 cases which we will insert things in the middle, or at the end or at the start. 
what are conditions for each of these

3 cases 
 mid -- head < insertVal < head.next           & head.next>head

first -- insertVal>head and insertVal>head.next & head.next<head
last -- insertVal<head and insertVal<head.next

Now even after all this if the .next reaches head that means we have completed a loop and reached back without addition. 
This is only possible when all elements are same in that case add anywhere



class Solution(object):
    def insert(self, head, insertVal):
        newNode=Node(insertVal)
        if not head:                ## edge case
            newNode.next=newNode
            return newNode
        
        node=head
        while node:
            if ((node.val<=insertVal and  insertVal<node.next.val) and node.next.val>node.val) or   ##for equality i add =, but its not needed on 2nd condition 
            ((node.val<=insertVal and  node.next.val<insertVal) and node.next.val<node.val) or 
            ((node.val>insertVal and  node.next.val>insertVal) and node.next.val<node.val):
                tmp=node.next
                node.next=newNode
                newNode.next=tmp
                return head
            
            elif node.next==head:       ## it doesnt coincide with the add at end case because end addition becomes true before this 
                print("its all same")
                break 
            node=node.next
                
            
        tmp=node.next           ## add anywhere
        node.next=newNode
        newNode.next=tmp
        
        return head
==============================================================================================================  
143. Reorder List

Find middle node
Reverse after that 
merge two lists

# 1->2->3->4

#     M
# 1>2>3>4>NULL


# 1   2      
# 1>2>4>3>NULL

# 1>2>1.next>2.next>NULL  

#    1
# 1->2->3->4->5->6
#    2
# 7->8->9->10->11->12->13


class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        
        if head==None or head.next==None:
            return head
        
        fast=head
        slow=head 
        while fast and fast.next:
            prevSlow=slow
            slow=slow.next
            fast=fast.next.next
        prevSlow.next=None
        mid=slow
        
        prev=None
        node=mid
        while node:
            temp=node.next
            node.next=prev
            prev=node
            node=temp
            
        ptr1=head
        ptr2=prev
        
        while ptr2:
            if ptr1==None:
                prev2.next=ptr2
                ptr2=ptr2.next
            else:
                temp=ptr1.next 
                temp2=ptr2.next 
                ptr1.next=ptr2
                ptr2.next=temp

                ptr1=temp
                prev2=ptr2
                ptr2=temp2
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++Group : ARRAY++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                 
                
        
950. Reveal Cards In Increasing Order
------------------------------------------------------------------------------------------
Done,Very tricky 
------------------------------------------------------------------------------------------
Logic 
1. I think if you just focus on putting the correct number at the correct place it will work 
2. People has used deque or simulation solutions. I dont know this. 

I just think the following 
0. sort it
1. When you start you have to start revealing, so even index have to be filled first . Now depending on odd/even you either stacked at last or revealed at last. If the length of prev deck is odd, reveal Flag will switch 
if reveal Flag is TRUE again start filling with even indexes else with odd indexes
so reveal True, even True -- fill 
reveal False, even False -- fill  (classic XOR gate)

3. Now how to keep track of indexes, 
while filling just start filling otherwise store the indexes in a list  
4. you will have to create a list to keep track of new list

class Solution(object):
    def deckRevealedIncreasing(self, deck):
        """
        :type deck: List[int]
        :rtype: List[int]
        """
        deck.sort()   ### in place sort 
        reveal=True   ### starting with reveal = True as we start revealing in the first place 
        
        ret=[float('inf')]*len(deck) ### this is needed because we are filling random indexes
        
        count=len(deck)
        
        currList=range(len(ret))
        newList=[]
        while count!=0:
            for i in range(len(currList)):
                if (i%2==0) == reveal:      ### classic XOR gate 
                    ret[currList[i]]=deck.pop(0) #### remember to fill at index and not at i 
                    count-=1
                else:
                    newList.append(currList[i])   #### remember to append index and not i 
            if len(currList)%2!=0:          ### being odd switches the reveal Flag 
                reveal=not(reveal)           ## dont put False here, you have to switch 
                
            currList=newList
            newList=[]   ## dont forget this
                
        return ret

==============================================================================================================  
1. Two Sum

############Learning 
1. nums.sort() ### will sort nums in place
2. a=nums; a.sort() #### will again sort both nums and a inplace
3. a=sorted(nums) !!!!!!!!!!!!!!!  this will not sort nums will sort a
4. a=nums.sort() ### a will be NONE !!!!!!!!!!
5. Remember we have to return array positions in the the inital array 

2 Methods:
    1. Sort and then use two pointers O(Nlogn)
    2. Use dictionary O(N)
    3. Can also use binary search O(Nlogn)

------------------------------------------------------------------------------------------
Done,missed the easy to miss 
----------------------------------------------------------------------------------------------------------------
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        a=sorted(nums)
        
        i=0
        j=len(a)-1
        
        
        while i<j:
            if a[i]+a[j]<target:
                i+=1
            elif a[i]+a[j]>target:
                j-=1
            else:
                break         ##### YOU CAN BREAK HERE ONLY BECAUSE EXACTLY ONE SOLUTION IS MENTIONED!! 
                              ##### OTHERWISE YOU WOULD WANT TO CONTINUE
        
        ans=[]
        for x in range(len(nums)):
            if nums[x]==a[i]:
                ans.append(x)
                break               ### easy to miss this
        
        for y in range(len(nums)-1,-1,-1):
            if nums[y]==a[j]:
                ans.append(y)
                break               ### easy to miss this
            
        return ans            
    
----------------------------------------------------------------------------------------------------------------

2 sum using dictionary
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
    
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dict1={}
        
        for i in range(len(nums)):
            if target-nums[i] in dict1:
                return [i,dict1[target-nums[i]]]
            else:
                dict1[nums[i]]=i
                
        return None

Note an assumption here: We know that there is ONY ONE UNIQUE SOLUTION to an array. 
That is why we BREAK after we reach there. If the question allowed multiple solutions WE WOULD NOT HAVE BROKEN



Group: Two sum 
============================================================================================================== 
167. Two Sum II - Input array is sorted
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------


class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        i=0
        j=len(numbers)-1
        
        while i<j:
            if numbers[i]+numbers[j]<target:
                i+=1
            elif numbers[i]+numbers[j]>target:
                j-=1
            else:
                break
        
        return [i+1,j+1]            

==============================================================================================================          
532. K-diff Pairs in an Array
------------------------------------------------------------------------------------------
1. Similar to 2Sum
2. K=0 is a special case and will need more code
3. Whenever a number is present in the dict it has been included in count already, if it is not present 
then two possibility up or down 
------------------------------------------------------------------------------------------
Done, Tricky
------------------------------------------------------------------------------------------
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        #3,1,4,1,5
        
        dict1={}
        count=0
        for i in range(len(nums)):
            if nums[i] not in dict1:
                if nums[i]-k in dict1:
                    count+=1
                if nums[i]+k in dict1:
                    count+=1
                dict1[nums[i]]=1
            elif k==0:
                dict1[nums[i]]+=1
                
                
        if k==0:
            for i in dict1.keys(): ###only get those elements for which we found dups, one pair for each
                if dict1[i]>1:
                    count+=1            

        return count

==============================================================================================================               
3 sum using dictionary

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        target=0
        ans=[]
        for i in range(len(nums)):
            dict1={}
            for j in range(i+1,len(nums),1):      ### see how we only check forward 
                if target-nums[i]-nums[j] in dict1:
                    b=sorted([nums[i],nums[j],target-nums[i]-nums[j]])
                    if b not in ans:     ### see this way to avoid dups 
                        ans.append(b)
                else:
                    dict1[nums[j]]=1
                    
                    
        return ans

Time complexity: O(N2)


3 sum Using sorting 

########## can also be solved using sorting and two pointers O(N2)
1.Note how we can use of tuples and dict to dedup lists 
2. Note how we DONT BREAK after finding target. 
------------------------------------------------------------------------------------------
Done,missed the easy to miss 
------------------------------------------------------------------------------------------------

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        target=0
        ans=[]
        

        for i in range(len(nums)):
            j=i+1
            k=len(nums)-1
            while j<k:
                if nums[j]+nums[k]>target-nums[i]:
                    k-=1
                elif nums[j]+nums[k]<target-nums[i]:
                    j+=1
                else:
                    b=sorted([nums[i],nums[j],target-nums[i]-nums[j]])
                    if b not in ans: 
                        ans.append(b)
                    j+=1       #### YOU HAD TO CONTINUE BECAUSE THERE ARE MULTIPLE SOLUTIONS 
                    k-=1       #### DONT FORGET TO UPDATE THESE POINTERS
                            
            
        return ans 
 
==============================================================================================================

121. Best Time to Buy and Sell Stock
This is a tricky question to think about    !!!!!
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------------
## basically maintain a curmin and a max profit counter and iterate right
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        maxProfit=0
        currMin=prices[0]
        
        for i in range(1,len(prices),1):
            currMin= min(currMin,prices[i])
            maxProfit=max(maxProfit,prices[i]-currMin)    
        
        return maxProfit     
==============================================================================================================
122. Best Time to Buy and Sell Stock II
here simply multiple transactions are allowed
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        totalProfit=0
        currMin=float("inf")
        for i in range(len(prices)):
            currMin=min(currMin,prices[i])
            profit=prices[i]-currMin 
            if profit>0:
                totalProfit+=profit ###take this profit by selling the stock 
                currMin=prices[i]  ###dont reset it to float inf because you can buy and this same stock again
        return totalProfit
------------------------------------------------------------------------------------------------
multiple transactions are allowed so you just take the profit at each step and add it up 
just ignore losses (when stock goes down from i to i+1 and add profits, this is simply your ans

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        #l -- ok
        #s -- ok 
        #e -- empty ok 
         #  -- single ok 
        profit=0
        for i in range(1,len(prices),1):
            if prices[i]>prices[i-1]:  ### we take profits with each increases 
                profit+=prices[i]-prices[i-1]
        return profit


                
    
==============================================================================================================

88. Merge Sorted Array
Tags: Array
Group: Merge Sorted Array
----------------------------------------------------
Simple solution using extra space. Two list merge standard way. 
----------------------------------------------------  
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------------
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        ptr1=0
        ptr2=0
        
        list1=[]
        
        while ptr2<=n-1 and ptr1<=m-1:
            if nums1[ptr1]<= nums2[ptr2]:
                list1.append(nums1[ptr1])
                ptr1+=1
            else:
                list1.append(nums2[ptr2])
                ptr2+=1
                
        
        while ptr1<=m-1:
            list1.append(nums1[ptr1])
            ptr1+=1
        
        while ptr2<=n-1:
            list1.append(nums2[ptr2])
            ptr2+=1
            
                
        
        
        for i in range(len(list1)):
            nums1[i]=list1[i]
----------------------------------------------------
Now if you dont want to use extra space this can be optimized by using three pointers and going over backwards 
----------------------------------------------------
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        
        i=m-1
        j=n-1
        k=m+n-1
        
        while i>=0 and j>=0:
            if nums1[i]>=nums2[j]:
                nums1[k]=nums1[i]
                i-=1
            else:
                nums1[k]=nums2[j]
                j-=1
            k-=1
        
        
        while j>=0:
            nums1[k]=nums2[j]
            j-=1
            k-=1
            
        ###### WE DO NOT NEED THIS SECTION!!!!!! for i 
    
Group: merge lists
==============================================================================================================
        
27. Remove element
Group: Base explorer method
What is base explorer?
Moving some numbers to the end 
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------

The first thing is we are not removing elements we are just moving them to the end. 

Think two pointers: 
----------------------------------------------------------------------------------------------------
1 is the BASE pointer which always stays at the STARTING of the series of numbers to be moved to the end. 
----------------------------------------------------------------------------------------------------
2nd in the EXPLORER pointer which looks for non- value numbers to swap.
Now this logic can be implemented in two ways:
Using a while loop (this is how you will think but edges will be difficult to handle)
Using a for loop (more elegant but difficult to come up with) 

For loop solution 
----------------------------------------------------
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        
        bas=0
        
        for exp in range(len(nums)):
            if nums[exp]!=val:
                nums[bas],nums[exp]=nums[exp],nums[bas]
                bas+=1   ## making sure that base is always at starting point ## of val series 
                         ## note we don't have to worry about explorer 
                         ## it moves automatically 
                
        return bas


While loop solution
----------------------------------------------------
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        # l
        # s
        # e ## empty None
            ## single w val
            ## single /w val 
            
        bas=0
        exp=0
        
        while exp<=len(nums)-1:
            if nums[bas]==val and nums[exp]!=val:
                nums[bas],nums[exp]=nums[exp],nums[bas]
            elif nums[bas]!=val:
                bas+=1
                exp+=1
            elif nums[bas]==val and nums[exp]==val:
                exp+=1
        return bas

Notes on while loop solution:
1. We are only checking if exp crosses over or not bas because expl always ahead of base
2. We wrote 4 cases then realized two cases are similar so reverted back to three
3. We start both bas and exp from 0 otherwise we will miss edge cases.
4. bas and exp are pointers not numbers 
tags: move zeroes, remove duplicates from sorted array
5. For the smarter solution realize there is only one case: 
when swapping happens only then base moves, otherwise base is fixed and explorer ALWAYS moves ahead. 
6. Even if base is not at the right place in the beginning some useless swaps bring it to the right place. This is clever. memorize. 

==============================================================================================================
283. move zeros 
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
Same question as remove element 
Copy code
Group: Base explorer method

==============================================================================================================
26. Remove Duplicates from Sorted Array
Group: Base explorer method 


in this question 
1. you cannot just return unique char length. because the underlying array has to be modified
2. dictionary popping fucks up because of indexing issues
3. This is a very clever question. Base Explorer with a twist 
Trick is that the array is SORTED and duplicates will be next to each other. 
SORTED IS CRUCIAL 

We use the base and explorer method . Explorer explores while base sits is the starting point of the duplicate series. 
When explorer finds a new value we simply copy in front of base and increment it. Just like above

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        bas=0
        exp=0
        
        while exp<=len(nums)-1:
            if nums[bas]==nums[exp]:
                exp+=1
            elif nums[bas]!=nums[exp]:
                bas+=1
                nums[bas]=nums[exp]
                exp+=1
        return bas+1
==============================================================================================================
905. Sort Array By Parity
Group: Base explorer method 

Standard base explorer question 


class Solution(object):
    def sortArrayByParity(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        
        #### Some numbers need to be moved to the end.  BASE EXPLORER!
        #### What numbers? odd numbers 
        #### Base should sit at the starting of the series of numbers to be moved to         
        #### the end (odd numbers here)
        
        bas=0
        exp=0
        while exp<=len(A)-1:
            if A[bas]%2!=0 and A[exp]%2!=0:
                exp+=1
            elif A[bas]%2!=0 and A[exp]%2==0:
                A[bas],A[exp]=A[exp],A[bas]
            elif A[bas]%2==0:
                exp+=1
                bas+=1
        return A
==============================================================================================================            
443. String Compression
Group: Base explorer method 
------------------------------------------------------------------------------------------
Done,slightly difficult due to cases not logic 
------------------------------------------------------------------------------------------
class Solution(object):
    def compress(self, chars):
        """
        :type chars: List[str]
        :rtype: int
        """
        bas=0
        exp=0
        count=0
        while exp<=len(chars)-1:
            if chars[bas]==chars[exp]:
                exp+=1
                count+=1
            elif count==1:
                chars[bas+1]=chars[exp]
                bas=bas+1
                count=0
            else:
                chars[bas+1:bas+1+len(str(count))]=list(str(count))      # This line works regardless of my doubts 
                chars[bas+1+len(str(count))]=chars[exp]
                bas=bas+1+len(str(count))
                count=0 
        if count==1:         #### missed this end case 
            return bas+1
        else:
            chars[bas+1:bas+1+len(str(count))]=list(str(count))
            return bas+1+len(str(count))             
==============================================================================================================
169. Majority Element
Given an array of size n, find the majority element. The majority element is the element that appears more than n/2 times.
You may assume that the array is non-empty and the majority element always exist in the array.
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
Simple just use a dict to maintain count and break whenever it exceeds. Didnt do Bayer moore algo             
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        dict1={}
        
        for i in range(len(nums)):
            if nums[i] not in dict1:
                dict1[nums[i]]=1
            else:
                dict1[nums[i]]+=1
            if dict1[nums[i]]>len(nums)/2:   ## greater than and not >= because majority is more than half 
                return nums[i]

==============================================================================================================              
349. Intersection of Two Arrays
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
just check if it is in other or not 


class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        ans=[]
        for i in range(len(nums1)):
            if nums1[i] in nums2 and nums1[i] not in ans:
                ans.append(nums1[i])
        return ans

Time complexity: O(n*m)



use a dict and set or two dicts: 
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        ans=[]
        dict1={}
        for i in range(len(nums1)):
            if nums1[i] not in dict1:
                dict1[nums1[i]]=1
        
        for j in range(len(nums2)):
            if nums2[j] in dict1:
                ans.append(nums2[j])
     
        return set(ans) 

Time complexity: O(n+m)



class Solution(object):
    def intersection(self, nums1, nums2):
        return set(nums1).intersection(set(nums2))

==============================================================================================================

189. Rotate Array
------------------------------------------------------------------------------------------
Done,Complexity constraint: solving in-place and in O(n). Do later
------------------------------------------------------------------------------------------
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        
        k=k%len(nums)
        
        A=nums[len(nums)-k:]+nums[:len(nums)-k]
        
        for i in range(len(A)):
            nums[i]=A[i]

Space copmplexity: O(n)


==============================================================================================================

485. Max Consecutive Ones
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
Simple question but i got tricked and tried to compare prev and next which isnt necessary
# simple iteration and keeping count 
class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        counter=0
        countermax=0
        for i in range(len(nums)):
            if nums[i]==1:
                counter+=1
                countermax=max(counter,countermax)
            else:
                counter=0
        
        return countermax

==============================================================================================================
682. Baseball Game
1.Maintaining a list is important 
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------
class Solution(object):
    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """
        sum1=0
        list1=[]
        for i in range(len(ops)):
            if ops[i]=="+":
                list1.append(list1[-1]+list1[-2])
                sum1+=int(list1[-1])
            elif ops[i]=="C":
                sum1-=int(list1[-1])
                list1.pop()
                
            elif ops[i]=="D":
                list1.append(2*list1[-1])
                sum1+=list1[-1]
                
            else:
                list1.append(int(ops[i]))
                sum1+=list1[-1]
                
        return sum1 

============================================================================================================
665. Non-decreasing Array
Tag: Peak, Valley question
1. you have to count violations or decreases, if second decreases happens return False
2. Now even if there is one violation, is it fixable ?? Its fixable only by doing two things 
Increase i to level of i-1 or decrease i-1 to level of i and check the neigbors after doing that. 
Again i-2 and i+1 wont exist at edges, use SAFROT here.


          i+1
 i-1     -
    -   -
   - - -
  -   -
 -    i
-
i-2
------------------------------------------------------------------------------------------
Done,very difficult
------------------------------------------------------------------------------------------

class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        count=0
        for i in range(1,len(nums),1):
            if nums[i]<nums[i-1]:
                count+=1
                if i!=1 and i!=len(nums)-1 and nums[i+1]<nums[i-1] and nums[i-2]>nums[i]:
                    return False
                if count>1:
                    return False                

        return True
==============================================================================================================
41. First Missing Positive
------------------------------------------------------------------------------------------
Done,difficult question 
------------------------------------------------------------------------------------------

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        #1. make sure 1 is not the missing number
        #2. Replace negatives with 1 and greater than n as 1 as they cant be the answer
        #3. Use indexes to store if a certain number is there, ### duplicates will mess up sign
        #4. Check what is negative, if nothing then return n+1
        
        for i in range(len(nums)):
            if nums[i]==1:
                break
        else:
            return 1
        
        # replace numbers greater than len(nums) and negatives as 1
        for i in range(len(nums)):
            if nums[i]<=0 or nums[i]>len(nums):
                nums[i]=1
        
        ## [3,4,1,1]
        ## [-3,4,-1,-1]
        for i in range(len(nums)):
            if nums[abs(nums[i])-1]>0:      ### duplicates will mess up sign so we change once
                nums[abs(nums[i])-1]=-1*nums[abs(nums[i])-1]  ### dont miss abs here
            
        for i in range(len(nums)):
            if nums[i]>0:
                return i+1
        return len(nums)+1
==============================================================================================================        
            
    
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++Group : GRID MANIPULATION++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                 


==============================================================================================================
463. Island Perimeter
class Solution(object):
    def islandPerimeter(self, grid):
        dirs=[[1,0],[-1,0],[0,1],[0,-1]]
        m=len(grid)
        n=len(grid[0])
        count=0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    neigbors=[[i+dir1[0],j+dir1[1]] for dir1 in dirs]
                    sides=4
                    for neighbor in neigbors:
                        nextX=neighbor[0]
                        nextY=neighbor[1]
                        if nextX>=0 and nextX<=m-1 and nextY>=0 and nextY<=n-1 and grid[nextX][nextY]==1:
                            sides-=1
                    count+=sides
        return count
## 4 sides land, 0 added
## 3 sides land, 1
## 2 side land, 2
## 1 side land,3
## 0 side land, 4 added
    
==============================================================================================================
48. Rotate Image
----
several possible ways to get the same thing : at least 4 but use horizontal swapping to make things easier 
---
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------

class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        m=len(matrix)
        n=len(matrix[0])
        
        for i in range(m):
            for j in range(i,n,1):
                matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
        
        for i in range(m):
            matrix[i][:]=matrix[i][::-1]
==============================================================================================================
832. Flipping an Image
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------

Simple horizontal swap and use "not" to reverse 

class Solution(object):
    def flipAndInvertImage(self, image):
        """
        :type image: List[List[int]]
        :rtype: List[List[int]]
        """
        
        for i in range(len(image)):
            image[i][:]=image[i][::-1]
        
        
        for i in range(len(image)):
            for j in range(len(image[0])):
                image[i][j]=int(not(image[i][j]))
                
        return image
==============================================================================================================        
Reshape the matrix
------------------------------------------------------------------------------------------
Done,
------------------------------------------------------------------------------------------

Using extra space


class Solution(object):
    def matrixReshape(self, nums, r, c):

        if len(nums)*len(nums[0])!=r*c:
            return nums
            
        list1=[]
        
        for i in range(len(nums)):
            for j in range(len(nums[0])):
                list1.append(nums[i][j])
        

        ans=[]
        x=0
        for i in range(r):            
            list2=[]
            for j in range(c):
                list2.append(list1[x])
                x+=1
            ans.append(list2)
            
        return ans

        OR
        mat2=[[0]*c for x in range(r)]       
        for i in range(r):
            for j in range(c):
                mat2[i][j]=list1[0]
                list1.pop(0)
        return mat2
------------------------------------------------------------------------------------------------
No extra space 
# rows and columns can be written as index/c and index%c

class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        if len(nums)*len(nums[0])!=r*c:
            return nums
            
        ans=[[0 for x in range(c)] for y in range(r)]
        
        for i in range(len(nums)):
            for j in range(len(nums[0])):
                ans[(i*len(nums[0])+j)/c][(i*len(nums[0])+j)%c]=nums[i][j]
                
        return ans
==============================================================================================================  
766. Toeplitz Matrix

I calculated equations of each diagonal line
which came out as i=j+k-n with k ranging from 1 to m+n-1, Now i generate points across each line and check if its equal

class Solution(object):
    def isToeplitzMatrix(self, matrix):
        
        m=len(matrix)
        n=len(matrix[0])
        
        for k in range(1,m+n,1):
            list1=[]
            for j in range(n):
                i=j+k-n
                if i>=0 and i<=m-1:
                    list1.append(matrix[i][j])
            if not all([x==list1[0] for x in list1]):
                return False
        return True
------------------------------------------------------------------------------------------------
Very Clever solution:
    just check i+1, j+1 at each point if it is same then we are good
class Solution(object):
    def isToeplitzMatrix(self, matrix):
        m=len(matrix)
        n=len(matrix[0])
        
        for i in range(m-1):
            for j in range(n-1):
                if matrix[i+1][j+1]!=matrix[i][j]:
                    return False
        return True
------------------------------------------------------------------------------------------------
Method3:
can use i-j differences to create dictionaries, same diagonal i-j will be same
==============================================================================================================       
498. Diagonal Traverse
Points 
# i,j 
# 0,0

# 0,1
# 1,0 

# 0,n-1 ## corner 
# 1,n-2

# 1,n-1
# 2,n-2

# m-2,n-1
# m-1,n-2

# m-1,n-1
Equations 
# i+j=0
# i+j=1
# .
# .
# i+j=n-1 ##corner
# i+j=n
# i+j=n+1
# .
# .
# i+j=m+n-2
Simply following line generating scheme 
class Solution(object):
    def findDiagonalOrder(self, mat):
        m=len(mat)
        n=len(mat[0])
        
        ans=[]
        for k in range(0,m+n-1,1):
            
            if k%2!=0:
                for i in range(m):
                    j=k-i
                    if j>=0 and j<=n-1:
                        ans.append(mat[i][j])
            else:
                for i in range(m-1,-1,-1):
                    j=k-i
                    if j>=0 and j<=n-1:
                        ans.append(mat[i][j])
                
                        
                        
        return ans

Another solution which is good and probably much faster
https://leetcode.com/problems/diagonal-traverse/discuss/581868/Easy-Python-NO-DIRECTION-CHECKING

==============================================================================================================       
73. Set Matrix Zeroes
------------------------------------------------------------------------------------------
Done, did first two methods do third later
------------------------------------------------------------------------------------------
The key point here is that we need to preserve original zeroes and separate them from cretaed zeroes


class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        
        m=len(matrix)
        n=len(matrix[0])
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==0:                    
                    for x in range(n):
                        if matrix[i][x]!=0:    ### you have to check for this very important
                            matrix[i][x]='h'
                    for x in range(m):
                        if matrix[x][j]!=0:
                            matrix[x][j]='h'  
                            
                            
        for i in range(m):
            for j in range(n):
                if matrix[i][j]=='h':
                    matrix[i][j]=0   


Space complexity : O(1)
------------------------------------------------------------------------------------------------
Create two extra arrays to store where the zeroes are 

class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        
        m=len(matrix)
        n=len(matrix[0])
        
        rows=[False for x in range(m)]
        columns=[False for x in range(n)]
        
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==0:
                    rows[i]=True
                    columns[j]=True
        
        for i in range(m):
            for j in range(n):
                if rows[i] or columns[j] ==True:
                    matrix[i][j]=0
        
        return matrix


Time Complexity: O(M*N) -- two passes over matrix 

Space Complexity: O(M + N)
------------------------------------------------------------------------------------------------
Instead of creating extra arrays try to use the matrix rows and columns itself. 
Now dealing with first row and first column becomes very tricky. Why? because it is difficult to differentiate between original zero and created zero.  
I use extra variables to manage these two lines. 

The order is very important. Otherwise this question gets real messy real fast. 
1. Store 0 in 1st row and 1st column. 
2. DONT DONT DONT ITERATE ON THESE TWO COLUMNS. CHANGE THE REST FIRST ACCORDING TO THE VALUES IN THESE TWO
3. I MARKED WHETHER OR NOT THESE TWO COLUMNS ARE ZERO USING EXTRA VARIABLES. ITERATE ON JUST THESE TWO AND MODIFY NOW.



class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        
        m=len(matrix)
        n=len(matrix[0])
        

        first_col=False
        first_row=False
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==0:
                    matrix[0][j]=0
                    matrix[i][0]=0
                    if i==0:                     #### Using extra flags for first column and first row 
                        first_row=True
                    if j==0:
                        first_col=True
        
        
        
        for i in range(1,m,1):   ### Now iterate only on the rest of matrix and leave aside 1st row and column. This is done not to mess up the orginal storage area 217. Contains Duplicate

            for j in range(1,n,1):
                if matrix[0][j]==0 or matrix[i][0]==0:
                    matrix[i][j]=0
        
        if first_col:                              
            for i in range(m):
                matrix[i][0]=0
        
        if first_row:
            for j in range(n):
                matrix[0][j]=0
                
        
        
        
        return matrix

Time complexity O(m*n)
Space complexity O(1)
==============================================================================================================
54. Spiral Matrix 

------------------------------------------------------------------------------------------
have to debug
------------------------------------------------------------------------------------------

1.Final solution list will contain (m*n) elements.
2.Use variables to track current row and current column. 
3.traverse in following sequence 
    left to right
    up to down
    Right to left
    Down to up
4. Use direction variable to keep track of above directions. Decide direction by current row number, current column number with total no. of rows and total no. of columns respectively.
5. Do not try to USE FOR LOOP HERE. Headache!! 
6. STUPID EDGE CASE OF ONLY COLUMN 

I spent a lot of time on this, trying to convert it into a double while loop and what not. Didnt work. Same pain. 
THE ORDER HERE IS PRINT FIRST and INCREMENT pointer first AND CHECK BOUNDARY CONDITION LATER. So we have effectively 
checked the boundary before we reach.
This makes every loop exclude the last element hence dealing with duplicates


class Solution(object):
    def spiralOrder(self, matrix):
        m = len(matrix)
        n = len(matrix[0])
        ans = []
        dir1 = 'right'
        iMax= m-1
        iMin= 0
        jMax= n-1
        jMin= 0
        i=0
        j=0
        
        if j==jMax:
            dir1='down'     ############### edge case of a single column
            
            
        while len(ans)<m*n:
            if dir1=='right':
                ans.append(matrix[i][j])
                j+=1
                if j==jMax:  ### when you hit the edge , change direction , i covered imin row , increment
                    iMin+=1                
                    dir1='down'
                    
            elif dir1=='down':
                ans.append(matrix[i][j])
                i+=1
                if i==iMax:
                    jMax-=1                
                    dir1='left'
            
            elif dir1=='left':
                ans.append(matrix[i][j])
                j-=1
                if j==jMin:
                    iMax-=1                
                    dir1='up'
                
            elif dir1=='up':
                ans.append(matrix[i][j])
                i-=1
                if i==iMin:
                    jMin+=1                
                    dir1='right'

                    
        return ans   
==============================================================================================================
Spiral Matrix II 

Same as spiral matrix 1 

class Solution(object):
    def generateMatrix(self, n):
        matrix1=[[0 for x in range(n)] for x in range(n)]
        maxi=n-1
        mini=0
        maxj=n-1
        minj=0
        
        dir1='right'
        
        i=0
        j=0
        
        for x in range(1,n*n+1,1):
            if dir1=='right':
                matrix1[i][j]=x
                j+=1
                if j==maxj:
                    mini+=1
                    dir1='down'
            elif dir1=='down':
                matrix1[i][j]=x
                i+=1
                if i==maxi:
                    maxj-=1
                    dir1='left'
            elif dir1=='left':
                matrix1[i][j]=x
                j-=1
                if j==minj:
                    maxi-=1
                    dir1='up'
            elif dir1=='up':
                matrix1[i][j]=x
                i-=1
                if i==mini:
                    minj+=1
                    dir1='right'
        return matrix1                                
==============================================================================================================
1275. Find Winner on a Tic Tac Toe Game

## Return winner - "A" or "B" -- 
## do we have a winner column or diagonal 
## only 8 ways, check all and return winner  
## elif no win if any empty -- pending
## else Draw
## Pending -- else case

class Solution(object):
    def tictactoe(self, moves):
        """
        :type moves: List[List[int]]
        :rtype: str
        """
        ## create and populate grid
        grid=[[" " for x in range(3)] for y in range(3)]
        for i in range(len(moves)):
            x=moves[i][0]
            y=moves[i][1]
            grid[x][y]="A" if i%2==0 else "B"

        emptyCounter=False
        eight=[ [[0,0],[0,1],[0,2]], 
                [[0,0],[1,0],[2,0]], 
                [[0,0],[1,1],[2,2]], 
                [[2,0],[1,1],[0,2]], 
                [[2,0],[2,1],[2,2]], 
                [[0,2],[1,2],[2,2]],
                [[1,0],[1,1],[1,2]],
                [[0,1],[1,1],[2,1]]]
        
        for i in range(len(eight)):
            list1=[]
            for j in range(3):
                x=eight[i][j][0]
                y=eight[i][j][1]
                x1=eight[i][0][0]
                y1=eight[i][0][1]
                if grid[x][y]==" ":
                    emptyCounter=True
                list1.append(grid[x][y])
            if all([x=="A" for x in list1]) or all([x=="B" for x in list1]):
                return list1[0]
        if emptyCounter:
            return "Pending"
        else:
            return "Draw"
------------------------------------------------------------------------------------------
Improvements 
instead of listing indices i can do row sum check and column sum checks, diag sum checks and use integers instead of A, B in grid
instead of empty counter i can simply see count(moves) for pending and draw

class Solution(object):
    def tictactoe(self, moves):
        ## create and populate grid
        grid=[[0 for x in range(3)] for y in range(3)]
        for i in range(len(moves)):
            x=moves[i][0]
            y=moves[i][1]
            grid[x][y]=1 if i%2==0 else -1

        #row sum check
        for i in range(3):
            if sum(grid[i])==3:
                return "A"
            elif sum(grid[i])==-3:
                return "B"
        
        #column sum check
        for j in range(3):
            jthsum=sum([row[j] for row in grid]) ### columns are not directly accessible, use this trick for column access
            if jthsum==3:
                return "A"
            elif jthsum==-3:
                return "B"
        #diag1 sum check
        sum1=0
        sum2=0
        for i in range(3):
            sum1+=grid[i][i]
            sum2+=grid[i][2-i]  ###n-1-i
            
        if sum1==3 or sum2==3:
            return "A"
        elif sum1==-3 or sum2==-3:
            return "B"
        
        if len(moves)<9:            ### now if the number of moves is less we can simply say pending
            return "Pending"    
        else:
            return "Draw"
==============================================================================================================
36. Valid Sudoku
0. We dont have to tell if the sudoku is SOLVABLE. Just have to comment on its current posiition 
1. n dicts for n row. n dicts for n columns. If repetition happens. We return False obviously
2. How to deal with the 9 boxes. Convert their original indices to row (0 to 3) and column (0 to 3) by dividing cordinates by 3. n dicts here too
3. Do i need multiple passes or can I update all dicts simulataneously in single pass. Single is good, update all dicts which passing

USE THIS WONDERFUL WAY OF PRINTING EVERYWHERE!!!! ALL YOUR PAIN IS GONE!!!!

class Solution(object):
    def isValidSudoku(self, board):
        logging=False
        n=9
        list_valid="123456789"
        list_column_hash=[{} for i in range(n)]     ### smart way of creating dicts
        list_row_hash=[{} for i in range(n)]
        list_box_hash=[[{} for i in range(n/3)] for j in range(n/3)]
        # if logging: print(list_column_hash)       ## SUPER SMART LOGGING
        # if logging: print(list_row_hash) 
        # if logging: print(list_box_hash) 
        for i in range(n):
            for j in range(n):
                if board[i][j] in list_valid:
                    #if logging : print(board[i][j])
                    ## column check 
                    if board[i][j] not in list_column_hash[j]:
                        list_column_hash[j][board[i][j]]=1
                    else:
                        return False
                    ## row check 
                    if board[i][j] not in list_row_hash[i]:
                        list_row_hash[i][board[i][j]]=1
                    else:
                        return False
                    ## box check
                    newI,newJ=i/(n/3),j/(n/3)
                    if board[i][j] not in list_box_hash[newI][newJ]:
                        list_box_hash[newI][newJ][board[i][j]]=1
                    else:
                        return False
                    # if logging: print(list_column_hash) 
                    # if logging: print(list_row_hash) 
                    # if logging: print(list_box_hash) 
        return True    
        
        

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++Group : STRINGS++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                 
392. Is Subsequence
------------------------------------------------------------------------------------------
done
------------------------------------------------------------------------------------------
2 pointers
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        i=0
        j=0
        
        while i<len(s) and j<len(t):    ### case when both pointers are inside
            if t[j]==s[i]:              ### We have done similar in the past in lists, arrays, link list etc 
                i+=1
                j+=1
            else:
                j+=1

    ### when any of the above two condition fails 
    ### we have our answer, if t is over and s is not then ans is False
    ### if s is over and t is not ans is true
    ### when both simulataneously over, ans is true 
        if i==len(s):
            return True
        else:
            return False

Do DP edit distance question
==============================================================================================================
387. First Unique Character in a String
------------------------------------------------------------------------------------------
done
------------------------------------------------------------------------------------------

Note the clever use of float("inf"). If multiple occurings occur you cant remove 




class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s=="":
            return -1
        
        dict1={}
        for i in range(len(s)):
            if s[i] in dict1:
                dict1[s[i]]=float('inf')
            else:
                dict1[s[i]]=i
                
        ans=min(dict1.values())
        
        if ans==float('inf'):
            return -1
        else:
            return ans 
==============================================================================================================
28. Implement strStr()
------------------------------------------------------------------------------------------
done
------------------------------------------------------------------------------------------

class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        
        if needle=='':
            return 0
        
        
        n=len(needle)
        
        for i in range(len(haystack)):
            if haystack[i:i+n]==needle:
                return i
        
        return -1
==============================================================================================================
151. Reverse Words in a String

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        list1=s.split()
        
        return " ".join(list1[::-1])
==============================================================================================================
468. Validate IP Address
------------------------------------------------------------------------------------------
done, pedantic tricky edge cases
------------------------------------------------------------------------------------------

class Solution(object):
    def ip4Check(self,input):
        allowed="0123456789"
        if input=="":
            return False
        for j in input:
            if j not in allowed:
                return False
        if int(input)>255 or int(input)<0:
            return False
        if input[0]=="0" and len(input)>1:
            return False
        return True
    
    def ip6Check(self,input):
        allowed="0123456789abcdef"
        if len(input)>4 or len(input)<1:
            return False
        for j in input:
            if j.lower() not in allowed:
                return False
        return True
        
        
    def validIPAddress(self, queryIP):
        """
        :type queryIP: str
        :rtype: str
        """
        list1=queryIP.split('.')
        list2=queryIP.split(':')
        
        if len(list1)==4:
            for i in list1:
                if not self.ip4Check(i):
                    return "Neither"
            return "IPv4"
                
                
        elif len(list2)==8:
            for i in list2:
                if not self.ip6Check(i):
                    return "Neither"
            return "IPv6"
        else:
            return "Neither"
==============================================================================================================
345. Reverse Vowels of a String


class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowels={'a':1,'e':1,'i':1,'o':1,'u':1,'A':1,
               'E':1,'I':1,'O':1,'U':1}
        i=0
        j=len(s)-1
        list1=list(s)
        
        
        while i<j:
            if list1[i] in vowels and list1[j] in vowels:
                list1[i],list1[j]=list1[j],list1[i]
                i+=1
                j-=1
            elif list1[j] not in vowels:
                j-=1
            elif list1[i] not in vowels:
                i+=1
        return ''.join(list1)
==============================================================================================================
125. Valid Palindrome
1. Strings cant be mutated using indexing
------------------------------------------------------------------------------------------
done, 
------------------------------------------------------------------------------------------
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        vowels="0123456789abcdefghijklmnopqrstuvwxyz"
        i=0
        j=len(s)-1
        while i<j:
            if s[i].lower() in vowels and s[j].lower() in vowels and s[i].lower()==s[j].lower():
                i+=1
                j-=1
            elif s[i].lower() not in vowels:
                i+=1 
            elif s[j].lower() not in vowels:
                j-=1
            else:
                return False
            
        return True
Space complexity: O(1) and Time Complexity: O(n)
------------------------------------------------------------------------------------------
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        vowels="0123456789abcdefghijklmnopqrstuvwxyz"
        a=''
        for i in range(len(s)):
            if s[i].lower() in vowels:
                a+=s[i].lower()
        
        return a==a[::-1] 
Space complexity: O(n) and Time Complexity: O(n)
==============================================================================================================
680. Valid Palindrome II
Once we have a violation we can either increase i or decrease j. The trick is you have to do both not just one
write the palindrome check twice

class Solution(object):
    def checkPalindrome(self,i,j,s):
        while i<=j:
            if s[i]==s[j]:
                i+=1
                j-=1
            else:
                return False
        return True
            
    
    def validPalindrome(self, s):
        i=0
        j=len(s)-1
        while i<=j:
            if s[i]==s[j]:
                i+=1
                j-=1
            else:
                return self.checkPalindrome(i,j-1,s) or self.checkPalindrome(i+1,j,s)
        return True

==============================================================================================================
58. Length of Last Word

class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        list1=s.split()
        return len(list1[-1])     
==============================================================================================================
14. Longest Common Prefix


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        ans=''
        for i in range(len(strs[0])):
            for j in range(len(strs)):
                if i>len(strs[j])-1 or strs[0][i]!=strs[j][i]: ### USED SAFROT (SKIP AND FALSE RUN OR TRUE)
                                                               ### logic for this condition
                    return ans 
            ans+=strs[0][i]
        return ans  ### this is for edge case
==============================================================================================================
242. Valid Anagram
------------------------------------------------------------------------------------------
Using two dicts
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        
        dict1={}
        dict2={}
        for i in range(len(s)):
            if s[i] not in dict1:
                dict1[s[i]]=1
            else:
                dict1[s[i]]+=1
        for i in range(len(t)):
            if t[i] not in dict2:
                dict2[t[i]]=1
            else:
                dict2[t[i]]+=1
        return dict1==dict2
------------------------------------------------------------------------------------------ 
Using 1 dict

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        
        dict1={}
        for i in range(len(s)):
            if s[i] not in dict1:
                dict1[s[i]]=1
            else:
                dict1[s[i]]+=1
                
        for i in range(len(t)):
            if t[i] not in dict1:
                return False
            elif dict1[t[i]]==1:
                del dict1[t[i]]
            else:
                dict1[t[i]]-=1
        return dict1=={}
------------------------------------------------------------------------------------------    
convert string to list and sort. Then compare!!!

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s=list(s)
        t=list(t)
        s.sort()
        t.sort()
        
        if len(s)!=len(t):             ### OR SIMPLY     return s==t!!!!!! after sorting
            return False
            
        
        for i in range(len(t)):
            if t[i]!=s[i]:
                return False
        return True
==============================================================================================================
49. Group Anagrams

Create a hash which notes frequency of each string. This hash will be same for strings in a group.
Dont miss the sorted dict keys while creating the hash.

Alternates
1. I am sorting the hash, but another way is to have a 26 places separated by "#" which will have frequency of each letter. 
2. We can also sort the anagrams so they are all the same. Time complexity increases.What is it ?

class Solution(object):
    def hasher(self,str1):
        dict1={}
        for i in range(len(str1)):
            if str1[i] not in dict1:
                dict1[str1[i]]=1
            else:
                dict1[str1[i]]+=1
        
        hash1=''
        for key in sorted(dict1.keys()):        ## is this sorting bad ??
            hash1+=key+str(dict1[key])
        return hash1
    
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        map1={}
        for i in range(len(strs)):
            hash1=self.hasher(strs[i])
            if hash1 in map1:
                map1[hash1].append(strs[i])
            else:
                map1[hash1]=[strs[i]] ## remember to use a list here
        
        ans =[]
        for key in map1.keys():
            ans.append(map1[key])
        return ans 

==============================================================================================================
249. Group Shifted Strings
The thing to note is shifted strings will have the same differences between i and i-1 chars. This can be used to generate a hash which will be common
for shifted strings

class Solution(object):
    def generate_hash(self,string):
        alphabets="abcdefghijklmnopqrstuvwxyz"
        alphabet_dict={alphabets[i]:i for i in range(len(alphabets))}
        ans=[]
        for i in range(1,len(string),1):
            delta=alphabet_dict[string[i]]-alphabet_dict[string[i-1]]
            ans.append(delta if delta>=0 else delta+26)
        return tuple(ans)
            
    def groupStrings(self, strings):
        dict1={}
        
        for i in range(len(strings)):
            hash1=self.generate_hash(strings[i])
            if hash1 not in dict1:
                dict1[hash1]=[strings[i]]
            else:
                dict1[hash1].append(strings[i])
        ans=[]
        for key in dict1.keys():
            ans.append(dict1[key])
        return ans
==============================================================================================================
408. Valid Word Abbreviation

class Solution(object):
    def validWordAbbreviation(self, word, abbr):
        n=len(abbr)
        m=len(word)
        alphabets="abcdefghijklmnopqrstuvwxyz"
        
        i=0
        j=0
        prevI=float("inf")
        while i<=n-1 and j<=m-1:
            if prevI==float("inf") and abbr[i] in alphabets: ### regular case where comparsion should happen
                if abbr[i]!=word[j]:
                    return False
                else:
                    pass
                    i+=1
                    j+=1
            elif prevI!=float("inf") and abbr[i] in alphabets:### numbr just completed on previous i
                number=int(abbr[prevI:i])
                if abbr[prevI:i][0]=="0": return False  ## 0 number wala case check 
                j+=number
                prevI=float("inf")
                    
            elif prevI!=float("inf") and abbr[i] not in alphabets:### numbr running
                i+=1
            elif prevI==float("inf") and abbr[i] not in alphabets: ### First time number found
                prevI=i
                
        
        if prevI!=float("inf"):           ## if it is at the last 
            number=int(abbr[prevI:i])
            if abbr[prevI:i][0]=="0": return False
            j+=number
            
        if j==m and i==n:
            return True
        else:
            return False
==============================================================================================================
722. Remove Comments
Very painful question because of the edge cases and in general
Lets remember a couple of things and hope for the best 
1. The comments are not just in the beginning of the line, yes you have to iterate and check
2. Yes you will have a blocked flag 
3. Its good to have while for inside loop because you need to jump around 
4. While we are blocked we are still searching for unblock and vice versa 
5. Anything before // or /* is needed so keep appending to a list 
6. Last condition is the trickiest where you append to ans only when you finally get unblocked

#Single line -ok 
# multiline -ok 
## merging
## // --ok you have to break the loop
class Solution(object):
    logging=True
    def removeComments(self, source):
        ans=[]
        blocked=False
        list1=[]
        for i in range(len(source)):
            j=0
            while j<=len(source[i])-1:
                if blocked:
                    ## searching for unblock
                    if source[i][j:j+2]=="*/":
                        blocked=False
                        j+=1    ##only increment by one because we increase below
                    
                else:
                    if source[i][j:j+2]=="/*":
                        blocked=True
                        j+=1
                    if source[i][j:j+2]=="//":
                        break
                    if not blocked:
                        list1.append(source[i][j])
                j+=1
            if list1 and not blocked: #### very tricky saw solution 
                ans.append(''.join(list1))
                list1 = []
        return ans



==============================================================================================================
++++++++++++++++++++++++++++++++++++++
+++++++Group : Design +++
++++++++++++++++++++++++++++++++++++++
706. Design HashMap
https://leetcode.com/problems/design-hashmap/discuss/1097755/JS-Python-Java-C%2B%2B-or-(Updated)-Hash-and-Array-Solutions-w-Explanation

Using a HUGE ARRAY AND directly using the value as a index in the array 
class MyHashMap:
    def __init__(self):
        self.data = [None] * 1000001 ###I create a huge array to store the dict possible 
                                     ### because there is a limit on key
    def put(self, key, val):    
        self.data[key] = val         ### I just use the value as index
                                     ### now array lookups are also O(1) as we dont search the whole array 
                                     ### just the index
    def get(self, key):
        val = self.data[key]         ### Retreival is also O(1)
        return val if val != None else -1
    def remove(self, key):
        self.data[key] = None       ### removal is also O(1)

Otherwise we use a hashing function for converting values to indexes.This can lead to collision so we add a linklist at each collision

Understand this solution later in detail 

class ListNode:         ##added another class
    def __init__(self, key, val, nxt):
        self.key = key  ## two special attributes
        self.val = val
        self.next = nxt 
class MyHashMap:
    def __init__(self):
        self.size = 19997
        self.mult = 12582917
        # No specific reason. For the size, I wanted something that was larger than the number of possible operations (10^4), but as small as possible without risking too many collisions, and preferably prime. The other is just a random large multiplier, also preferably a prime.

        self.data = [None for _ in range(self.size)] ## again we create a huge array
    def hash(self, key):
        return key * self.mult % self.size     ### modulo operator
    def put(self, key, val):
        self.remove(key)
        h = self.hash(key)
        node = ListNode(key, val, self.data[h])
        self.data[h] = node
    def get(self, key):
        h = self.hash(key)
        node = self.data[h]
        while node:
            if node.key == key: return node.val
            node = node.next
        return -1
    def remove(self, key):
        h = self.hash(key)
        node = self.data[h]
        if not node: return
        if node.key == key:
            self.data[h] = node.next
            return
        while node.next:
            if node.next.key == key:
                node.next = node.next.next
                return
            node = node.next
==============================================================================================================
380. Insert Delete GetRandom O(1)
https://leetcode.com/problems/insert-delete-getrandom-o1/solution/
Great article read it.
Hashmp+Array
Getrandom is dufficult in map so we maintain a list of keys to achieve that
Array+Hashmap

class RandomizedSet(object):
    def __init__(self):
        self.dict1={}
        self.list1=[]
    def insert(self, val):
        if val in self.dict1:
            return False
        self.list1.append(val)
        self.dict1[val]=len(self.list1)-1
        return True
    def remove(self, val):
        if val in self.dict1:
            ## swap the last number from the number we want to pop
            self.dict1[self.list1[-1]]=self.dict1[val] ## you have to correct indexes before swapping
            self.list1[self.dict1[val]],self.list1[-1]=self.list1[-1],self.list1[self.dict1[val]]
            
            self.list1.pop()
            del self.dict1[val]
            return True
        return False
    def getRandom(self):
        return random.choice(self.list1)     
==============================================================================================================
1229. Meeting Scheduler
LRU CACHE 
SPARSE MATRIX MULTIPLICATION



++++++++++++++++++++++++
Miscanellenous 
++++++++++++++++++++++++
==============================================================================================================
1507. Reformat Date


class Solution(object):
    def reformatDate(self, date):
        monthDict= {"Jan":"01", "Feb":"02", "Mar":"03", "Apr":"04", "May":"05", "Jun":"06", "Jul":"07", "Aug":"08", "Sep":"09", "Oct":"10", "Nov":"11", "Dec":"12"}
        date=date.split(" ")
        
        if len(date[0][:-2])==1:
            date1="0"+date[0][:-2]
        else:
            date1=date[0][:-2]
            
        year=date[2]
        month=monthDict[date[1]]
        ans="-".join([year,month,date1])
        
        return ans  

==============================================================================================================


==============================================================================================================
811. Subdomain Visit Count
# 1. Brute force
# any order of output

# ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
# 900 google.mail.com
# 900 mail.com
# 900 com
# 50 yahoo.com
# 50 com

# ## hashmap 
# 2. Optimised approach
# No optimization possible 

# 3. Regular Test case
# Done

# 4. Edge Cases
# Empty cpdomains
# Empty ""

# 5. Syntax
#    Checked


class Solution(object):
    def subdomainVisits(self, cpdomains):
        """
        :type cpdomains: List[str]
        :rtype: List[str]
        """
        domainCounts = {}
        # cpdomains: ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]

        for i in range(len(cpdomains)):
            count,domain = cpdomains[i].split(" ")
            while domain!="":
                if domain not in domainCounts:
                    domainCounts[domain]=int(count)
                else:
                    domainCounts[domain]+=int(count)
                domain=".".join(domain.split(".")[1:])
        
        ret=[]
        for key in domainCounts.keys():
            ret.append(str(domainCounts[key])+" "+key)
            
        return ret                    
==============================================================================================================

==============================================================================================================
13. Roman to Integer
We simply go back wards and add with one special case, we subtract when it decreases. (IV, IX)
class Solution(object):
    def romanToInt(self, s):
        dict1={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        ans=0
        for i in range(len(s)-1,-1,-1):
            if i+1<=len(s)-1 and dict1[s[i+1]]>dict1[s[i]]: ##SAFROT
                ans-=dict1[s[i]]
            else:
                ans+=dict1[s[i]]
        return ans
==============================================================================================================
12. Integer to Roman. Figured it myself :) Its not that difficult
### what is the largest you can divide by 
### and then next largest? keep on going 
## 
# 500:D Remainder=0
# 501:D Remainder=1 DI
# 499:ID
# 400:CD
# 49: 
# how are special cases of subtraction handled? just add them to the list nothing special
1. We keep on dividing by our biggest possible number where quotient is greater than or equal to 1.
2. We append the symbol* quotient into answer, not just symbol. Ex 3 is I*3 not just I
3. while condition is num>=1 because 1 is the biggest base. 
4. Thats it after that I just added the reduction cases in the list too and it magically worked.

class Solution(object):
    def intToRoman(self, num):
        dict1=[(1,'I'),(4,'IV'),(5,'V'),(9,'IX'),(10,'X'),(40,'XL'),(50,'L'),(90,'XC'),(100,'C'),(400,'CD'),(500,'D'),(900,'CM'),(1000,'M')]
        
        ans=''
        while num>=1:
            for i in range(len(dict1)-1,-1,-1):
                if num/dict1[i][0]>=1:   
                    ans+=dict1[i][1]*(num/dict1[i][0]) ##integer division makes sure we have ints
                    print(ans)
                    num=num%dict1[i][0]
                    print("num",num)
                    break
        
        return ans
        



==============================================================================================================
359. Logger Rate Limiter
Simply add to dictionary while printing and check if it is in dictionary and was printed before 10 seconds
class Logger(object):

    def __init__(self):
        self.dict1={}
        

    def shouldPrintMessage(self, timestamp, message):
        if (message not in self.dict1) or (message in self.dict1 and self.dict1[message]+10<=timestamp):
            self.dict1[message]=timestamp
            return True
        else:
            return False
==============================================================================================================
217. Contains Duplicate

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        dict1={}
        
        for i in range(len(nums)):
            if nums[i] in dict1:
                return True
            else:
                dict1[nums[i]]=1
        
        
        return False
==============================================================================================================
238. Product of Array Except Self
2 extra lists

class Solution(object):
    def productExceptSelf(self, nums):
        l=[0 for i in range(len(nums))]
        for i in range(len(nums)):
            if i==0:
                l[i]=1
            else:
                l[i]=l[i-1]*nums[i-1]
        r=[0 for i in range(len(nums))]    
        for i in range(len(nums)-1,-1,-1):
            if i==len(nums)-1:
                r[i]=1
            else:
                r[i]=r[i+1]*nums[i+1]
        return [r[i]*l[i] for i in range(len(nums))]
Space complexity O(N)
------------------------------------------------------------------------------------------           
Follow up with space complexity O(1)
class Solution(object):
    def productExceptSelf(self, nums):
        l=[0 for i in range(len(nums))]
        for i in range(len(nums)):
            if i==0:
                l[i]=1
                prev=1
            else:
                l[i]=nums[i-1]*prev
                prev=nums[i-1]*prev ## prev is a counter which is keeping track of entrire product to the left
        print(l)
        for i in range(len(nums)-1,-1,-1):
            if i==len(nums)-1:
                prev=1
            else:
                l[i]=l[i]*(nums[i+1]*prev)        
                prev=nums[i+1]*prev ## prev is a counter which is keeping track of entrire product to the right
        return l
------------------------------------------------------------------------------------------           
With division but handling 0 cases 
1. No zeros simply calculate product and divide
2. One zero, zero everywhere except at the zero. Still need to calculate product
3. Two zero, put zero everywhere
------------------------------------------------------------------------------------------    
class Solution(object):
    def productExceptSelf(self, nums):
        product=1
        countZero=0
        for i in range(len(nums)):
            if nums[i]==0:
                countZero+=1
            else:
                product=product*nums[i]
            
        if countZero==0:
            return [product/nums[i] for i in range(len(nums))]
        elif countZero==1:
            return [product if nums[i]==0 else 0 for i in range(len(nums))]
        elif countZero>1:
            return [0 for i in range(len(nums))]
------------------------------------------------------------------------------------------    
==============================================================================================================
Encode and Decode Strings
Stupid question honestly, there is no "encoding" as such
1. If we use (count of string+ #) it works well 
2. We can also use non-asccii delimiters
3. Double any hashes inside the strings, then use standalone hashes (surrounded by spaces) to mark string endings. For example:
{'abc', '#def'}   =>  "abc # ##def # "
'abc # ' -> 'abc ## ' so the insides are still not included

class Codec:
    def encode(self, strs):
        ans=''
        for i in strs:
            ans+=str(len(i))+"#"+i    
        return ans 
    def decode(self, s):
        nums='0123456789'
        list1=[]
        i=0
        while i<=len(s)-1:
            number=''
            while s[i]!="#":
                number+=s[i]
                i+=1
            list1.append(s[i+1:i+int(number)+1])
            i=i+int(number)+1
        return list1
==============================================================================================================
128. Longest Consecutive Sequence
1. Sorting ofcourse. We dont consider the dups to break the sequence nor extend it. 
O(nlogn)
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        #nums=list(set(nums))
        nums.sort()
        
        
        count=1
        countMax=1
        for i in range(1,len(nums),1):
            if nums[i]-nums[i-1]==1:
                count+=1
                countMax=max(countMax,count)
            elif nums[i]==nums[i-1]:
                pass
            else:
                count=1
                
        return countMax
 ------------------------------------------------------------------------------------------    
 Dictionary approach
 1. We use dictionary to store all so that we can lookup in o1 time
 2. Now we only go on a streak when its the first element. start streak only when n-1 doesnt exist 
Why is this O(N) time ? 
Looks like O(N2) to me because we go through the array and at each possible array we can start a streak which goes till N(max length of array)
But the thing is streak entry doesnt happen everywhere. 
That means, for example, 6,5,4,3,2,1 input, only the value 1 is valid for the loop(all other values have its value - 1 in the set), that is O(n).
Another corner example, 2, 5, 6, 7, 9, 11. All of these numbers are the "entrance" for the logic but the while loop doesn't run much. That is O(n) as well.


  class Solution(object):
    def longestConsecutive(self, nums):
        #nums=list(set(nums))
        if not nums:
            return 0
        dict1=collections.Counter(nums)
        count=1
        maxcount=1
        for i in range(len(nums)):  ### checking for every element in the list
            number=nums[i]
            
            if number-1 not in dict1: ### very clever and crucial, start streak only when n-1 doesnt exist   
                count=1
                while number+1 in dict1:
                    count+=1
                    maxcount=max(count,maxcount) 
                    del dict1[number]  ### THIS HAS AN EFFECT OF SPEEDING UP BECAUSE FUTURE STREAKS ARE PREVENTED      
                    #### DEL DICTIONARY IS EQUIVALENT TO USING SET TO DEDUP EARLIER
                    #### DUPLICATES CAUSE US TO GO ON THE SAME STREAK AGAIN AND AGAIN. THIS KEEP CUTTING THE LINK
                    #### WHILE WE MOVE TO THE NEXT
                    number+=1
                ## del dict1[number] ## THIS WILL NOT SPEEDEN UP BECAUSE THIS TAKES CARE OF ONLY ONE
        return maxcount
------------------------------------------------------------------------------------------    
You can also see this is a graph. Graph is unconnected and in several pieces. you need to know the length of longest piece.
Similar to Number of Islands? No of islands calculates no of unconnected components. Here we need to count nodes in each piece
Index arent the nodes here the numbers themselves are

class Solution(object):
    def dfs(self,number,dict1):
        count=0
        stack=[number]
        while stack:
            current=stack.pop()
            #print(current)
            neighbors=[current+1,current-1]
            for neighbor in neighbors:
                if neighbor in dict1 and dict1[neighbor]!='v':
                    stack.append(neighbor)
            count+=1
            dict1[current]='v'
        return count
    
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        #nums=list(set(nums))
        dict1=collections.Counter(nums)
        ### do dfs at every point if its not visited already 
        countMax=0
        for i in range(len(nums)):
            if nums[i] in dict1:
                count=self.dfs(nums[i],dict1)
                countMax=max(count,countMax)
        return countMax
------------------------------------------------------------------------------------------    
Same dfs with visited and one level opti
def dfs(self,number,dict1):
        count=1
        stack=[number]
        dict1[number]='v'
        while stack:
            current=stack.pop()
            neighbors=[current+1,current-1]
            for neighbor in neighbors:
                if neighbor in dict1 and dict1[neighbor]!='v':
                    dict1[neighbor]='v'
                    stack.append(neighbor)
                    count+=1
                    #print("neighbor",neighbor)
                    #print("count",count)
            
        return count
==============================================================================================================
Merge Triplets to Form Target Triplet
2 things:
1. Obviously we need the target numbers in the columns. We arent manufacturing numbers here, just taking maxes so if they dont exist then False
2. Now we only have max operators, so if any element is greater than corresponding target element, once we do operation, no way to reduce it back.
3. So based on these ideas, three Flags for three target elements. Iterate through all of them and make sure ALL flags are turned into True. 
A Flag can only turn True it is coming from an array which doesnt have ANY element greater than target elements and ofcourse, it should have the corresponding number. 

First solution in 8 mins 
class Solution(object):
    def mergeTriplets(self, triplets, target):
        flag=[False, False, False]
        
        for i in range(len(triplets)):
            tripletValidFlag=True
            for j in range(3):
                if triplets[i][j]>target[j]:
                    tripletValidFlag=False
                    break
            if tripletValidFlag:
                for j in range(3):
                    if triplets[i][j]==target[j]:
                        flag[j]=True
        
        if all(flag):
            return True
        else:
            return False
==============================================================================================================
763. Partition Labels
The question is asking for "as many partitions as possible". So we need to partition greedily as soon as possible. This will automatically result in 
most partitions. We cant partition at a given index if there is some char before the partition which has a repittion after the partition.
1. So we go through the array once and create a RIGHT MOST FOUND INDEX.
2. Now we use one iterating pointer to iterate the string, and maintain the second pointer for partitioning and one for base,  we keep updating  partitioning as we move. and move reset both base and partition after a split is made.

class Solution(object):
    def partitionLabels(self, s):
        rightMost={}
        for i in range(len(s)):
            rightMost[s[i]]=i
        
        ans=[]
        i=0
        partition=0
        base=0
        while i<=len(s)-1:
            if rightMost[s[i]]>partition:
                partition=rightMost[s[i]]
            elif i==partition:
                ans.append(partition-base+1)
                base=partition+1
                partition+=1
                
            i+=1
        
        return ans 
==============================================================================================================
Questions I have to think more about 

373. Find K Pairs with Smallest Sums
It does look like a 2 pointer question.
I imagined a 4 pointer solution to this question. Does that not work? Tried this.
TOO FUCKING COMPLICATED because you know what ? 4 pointers. Dont think like this lol







==============================================================================================================


==============================================================================================================
Questions where I know the logic but havent coded yet 





==============================================================================================================
Pathway: Strings, Array, Grid,Recursion,Tree, Graph,BinarySearch, DP,LinkList
        ,Stack,,Heap,Math 
        
Lists
=======================
https://neetcode.io/
https://www.lintcode.com/problem/?typeId=8

Blind 75 
Neetcode 150
https://leetcode.com/discuss/career/449135/how-to-effectively-use-leetcode-to-prepare-for-interviews
https://github.com/xizhengszhang/Leetcode_company_frequency
pramp.
https://python.plainenglish.io/python-for-interviewing-an-overview-of-the-core-data-structures-666abdf8b698
https://github.com/neetcode-gh/leetcode

Next 
Hand of Straights
Reorder list
LRU Cache
Container with most water

More_leetcode playlist
walmart playlist


================
0. APPLE LIST COMPLETE
======================
Peeking Iterator
274. H-Index
Sparse Matrix Multiplication
Median of Two Sorted Arrays ------------------
Trapping Rain Water     --------------------
Find Median from Data Stream ----------------------
Serialize and Deserialize N-ary Tree
Merge k Sorted Lists
Logical OR of Two Binary Grids Represented as Quad-Trees
Making File Names Unique
Smallest Integer Divisible by K
Smallest Rotation with Highest Score


1. REVISE AND MEMORIZE ALL PATTERNS 
==================================
2. REVISE ATTEMPT/BLIND 75
3. REVISE BLIND 150
4. READ DEEP LEARNING

======================================================================================================

Coding Style in interviews
1. Test cases - Come up with test cases at the beginning of the solution phase. Make sure that you have a solid understanding of the question, and you have finalized it with the interviewer. 
2. Test for edge cases. 
2. EXPLAIN BRUTE FORCE!!!
2. Explain your approach and only then begin coding 

2. Coding - create a structure of the program before attacking it. Make sure that the interviewer can see the structure. Or introduce them before what you are planning to do. 

3. Dry run - leave enough time for a dry run at the end. Don't assume that you will be able to click on the run button. Dry run itself is an art, and you get better with more practice. (commenting inline what the value is of a variable, taking care of recursive cases etc.) Ask more leading questions - The interviewer is on your side. They want to give you hints, but they are also looking for prompts where they can do so. Help them by asking more question. Some good examples - "Am I doing this right?" "Is this a good approach" "Do you think I could do any better" "Does this section of the code clear to you?" "Can I make it simpler?"

# 1. Brute force
# any order of output

# ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
# 900 google.mail.com
# 900 mail.com
# 900 com
# 50 yahoo.com
# 50 com

# Time Complexity:
# Space Complexity:     
# ## hashmap 
# 2. Optimised approach
# No optimization possible 

# 3. Regular Test case Dry run
# Done

# 4. Edge Cases
# Empty cpdomains
# Empty ""

# 5. Syntax
#    Checked

# 6. Readability
==============================================================================================================
ML
I was asked to code the logistic regression without using any external library like scikit-learn or scipy. I was to asked to code entirely in numpy
==============================================================================================================
CAREER PROGRESSION


Super high Payers
 ------------------------
 Roblox,Netflix, Stitch Fix, Instacart , Stripe, Snap, Airbnb

https://www.levels.fyi/comp.html?track=Software%20Engineer&title=Machine%20Learning%20Engineer&region=819
Which companies are paying a lot?
Apple, Snap

Apple Leveling 

Facebook Leveling

Snap Leveling




What do Amazon Sr DS, RS and AS do after Amazon. 
Microsoft?

Roles to Target:
DS to Research Data Scientist@Meta
AS to MLE@FB 
Data Scientist/Machine Learning Scientist to MLE@FB
AS to DS@TikTok
DS at Amazon to Senior DS@Airbnb
AS2 to Senior ML Scientist@Doordash
Ds2 to Staff DS,Walmart
DS2 to Senior RDS@Meta
Yashas DS to Senior DS@CVS to DS@Meta
DS to Senior DS@FB to Senior Applied Scientist@Uber
Senior Research Scientist to Sr. Data Scientist @google
AS to Staff Data Scientist@Sofi

MLE roles are available at tons of companies. I make 200k now. 
In one year its okay to go from hear to 350k something. I think this is the best I can do. 
Find out how much roles pay at different companies. 

1. You just need to know your shit for interviews
2. You need to perform on the job
==============================================================================================================
++++++++++++ OFFER NUMBERS MEGATHREAD ++++++++++++++++++
++++++++++++ COME HERE FOR MOTIVATION ++++++++++++++++++
https://www.levels.fyi/leaderboard/Data-Scientist/All-Levels/country/United-States/

Instacart, Facebook
1.Currently a Sr. DS at Microsoft(Analytics). Current TC: $250k 
Meta TC: $330k (Base: $180k/Annual Bonus: 15%/ Equity: $380k/ Sign-on: $25k) - IC5
Instacart: $456k (Base: $211k/Equity: $960k/Sign-on: $25k)- IC6
2. Instacart : 210K/916K/20K Level -- L6 data science
3. Instacart: 
Recruiter mentioned statistics/experimentation and problem solving interviews. I am thinking more of AB testing.

Stripe
Levels: L1,L2,L3 
Stripe L3 DS: $386,000

Snap
https://www.levels.fyi/company/Snap/salaries/Data-Scientist/










==============================================================================================================
++++++++ MACHINE LEARNING/MACHINE LEARNING DESIGN ++++++++++
1. Random Forest in depth - ESLR, read sklearn randomForest , 
2. Boosting in depth - ESLR
3. Gradient descent for common algo in depth
4. Serving deep learning chapter
 ------------------------------------------------------------
1. Stanford Course: https://stanford-cs329s.github.io/
2. https://huyenchip.com/ml-interviews-book/
3. https://huyenchip.com/archive/ -- ALL LINKS HERE
4. https://leetcode.com/discuss/interview-question/system-design/566057/machine-learning-system-design-a-framework-for-the-interview-day

More applied DEEP Learning
4. Good Deep Learning Course: https://course.fast.ai/
5. https://fullstackdeeplearning.com/
6. Dive into Deep Learning book -- http://d2l.ai/
7. 3Blue1Brown Playlist on DL
8. https://github.com/khangich/machine-learning-interview


Blogs:
Airbnb ( https://medium.com/airbnb-engineering/tagged/machine-learning )
Amazon?
Deepmind ( https://deepmind.com/blog )
Facebook ( https://ai.facebook.com/blog/ )
Google ( https://cloud.google.com/blog/products/ai-machine-learning, https://ai.googleblog.com/, https://www.blog.google/technology/ai/,
Linkedin (https://engineering.linkedin.com/blog/topic/machine-learning)
Netflix ( https://research.netflix.com/research-area/machine-learning )
Pinterest ( https://medium.com/pinterest-engineering/machine-learning/home, https://labs.pinterest.com/projects/machine-learning/ )
Quora (https://www.quora.com/q/quoraengineering/Machine-Learning-at-Quora)
Stripe ? ( https://stripe.com/blog/engineering/ ),
Twitter ( https://blog.twitter.com/engineering/en_us/topics/insights.html )
Uber ( https://eng.uber.com/research/?_sft_category=research-ai-ml )
https://mlengineer.io/
https://developers.google.com/machine-learning/recommendation/content-based/summary


Papers
https://www.microsoft.com/en-us/research/uploads/prod/2019/03/amershi-icse-2019_Software_Engineering_for_Machine_Learning.pdf
https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf )
More papers: https://www.teamblind.com/post/ML-design-interview-3cYD0vdM



Paid Websites
https://mlexpert.io/exercises
https://www.mle-interviews.com/
https://www.educative.io/courses/grokking-the-machine-learning-interview
Pham An Khang -- https://mlengineer.io/


Source:
https://www.teamblind.com/post/Machine-learning-engineering-and-ML-systems-design-resources-master-list-gWY7ZUTT
https://leetcode.com/discuss/interview-question/system-design/566057/machine-learning-system-design-a-framework-for-the-interview-day
https://leanpub.com/MLE
https://leanpub.com/theMLbook
https://www.amazon.science/latest-news/new-hands-on-guide-demonstrates-how-to-implement-natural-language-processing-business-solutions

https://mlengineer.io/from-a-so-so-swe-to-e5-meta-6bf37208657c
==============================================================================================================
++++SYSTEM DESIGN++++
Just resources here for now
https://www.teamblind.com/post/My-Approach-to-System-Design-V4SJARdx





==============================================================================================================
++++SQL++++
https://daetama.io/





        
        
        
        
               
            
        
        
        
        
                    
                
            
            
        
        
        
            
            
                
            
            

        
        
    


        
            
        
            
               



        
        
        








