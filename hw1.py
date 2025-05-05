#!/usr/bin/env python
# coding: utf-8

# ## algorithm design and anlysis-2025 spring  homework 1 
# **Deadline**：2025.5.14
# 
# **name**:
# 
# 
# note：
# ---
# 1. 带有\*的题目，申请免上课的同学，必须完成，其他同学选作；
# 2. 请独立完成，如求助了他人或者大模型，请著明，并且不可省略算法分析部分；
# 4. 如若作答有雷同，全部取消成绩；
# 3. 需要书面作答的题目，可以通过引用图片的形式添加，但是注意上传项目时包含所引用的图片的源文件；
# 4. $log_n$ 默认表示$log_2{n}$;

# ## 问题 1
# 
# 对于下面的每一对表达式(A, B), A是否能表示为B的 $\Theta, \Omega ,O$形式. 请注意, 这些关系中的零个、一个或多个可能成立。列出所有正确的。经常发生一些学生会,把指示写错, 所以请把关系写完整, 例如: $A = O(B),  A =\Theta(B)$, 或$A = \Omega(B)$。
# 
# 1. $A=n^2-100n, B=n^2$
# 2. $A=logn, B=log_{1.2}n$
# 3. $A=3^{2n}, B=2^{4n}$
# 4. $A=2^{logn}, B=n$
# 5. $A=\log{\log}{n},B=10^{10^{100}}$

# can refer a handwritten picture, pleas upload the picture in /fig/xxx.png
# answer:
# 

# ## 问题 2：
# 
# 假设有函数 $f$ 和 $g$ 使得 $f(n)$ = $O(g(n))$ 对于下面的每一个陈述, 请判断对错, 如果正确请给出证明, 否则请给出一个反例。
# 
# 1. $\log{f(n)}$ = $O(\log(1+g(n)))$
# 2. $3^{f(n)}=O(3^{g(n)})$
# 3. $(f(n))^2=O((g(n))^2)$ 

# you can refer a handwritten picture, pleas upload the picture in /fig/xxx.png
# answer:

# ## 问题 3
# 
# 根据下列递归公式, 计算下列 $T(n)$ 对应的的渐近上界。要求所求的边界尽可能的紧（tight）, 请写明步骤。
# 
# 1. $T(1)=1; T(n)=T(n/4)+1$ for $n>1$
# 2. $T(1)=1;T(n)=3T(n/3)+n^2$ for $n>1$
# 3. $T(1)=1;T(n)=T(2n/3)+1$ for $n>1$
# 4. $T(1)=1;T(n)=5T(n/4)+n$ for $n>1$
# 5. $T(n)=1 \ for\ n \le 2 ; T(n)=T(\sqrt{n})+1 \ for \ n>2$

# can refer a handwritten picture, pleas upload the picture in /fig/xxx.png
# answer:

# ## 问题 4：
# 
# 给定一个包含n个元素的数组 `profits` , 它的第 `i` 个元素 `profits[i]` 表示一支股票第 `i` 天的**收益**（正数表示涨, 负数表示跌）。你只能选择 **某一天** 买入这只股票, 并选择在 **未来的某一个不同的日子** 卖出该股票。
# 
# 1. 设计一个算法来计算你所能获取的最大利润和对应买入和卖出的日期。请分析算法方案, 计算其时间复杂度, 并且使用python编程实现该算法。
# 
# 2. \* 设计一个时间复杂度为 $O(n)$的算法实现该算法
# 
# e.g. :
# ---
# profits=[3,2,1,-7,5,2,-1,3,-1], 第5天买入, 第8天卖出, 收益最大：9
# 
# 

# idea:

# In[ ]:


def max_profit(profits): #时间复杂度为O(n2)
    max_profit = float('-inf')
    best_start = 0
    best_end = 0
    n = len(profits)
    for i in range(n):
        for j in range(i, n):
            current_profit = sum(profits[i:j + 1])
            if current_profit > max_profit:
                max_profit = current_profit
                best_start = i
                best_end = j
    return (max_profit, best_start + 1, best_end + 1)


profits = input("输入利润: ").split()
profits = [float(x) for x in profits]
max_profit, buy, sell = max_profit(profits)
print(f"最大利润: {max_profit}, 买: {buy}, 卖: {sell}")


# In[ ]:


def max_profit(profits):
    max_so_far = max_now = profits[0]
    start = end = 0
    best_start = best_end = 0

    for i in range(1, len(profits)):
        if profits[i] > max_now + profits[i]:
            max_now = profits[i]
            start = end = i
        else:
            max_now += profits[i]
            end = i

        if max_now > max_so_far:
            max_so_far = max_now
            best_start = start
            best_end = end

    if best_start == best_end:
        max_pair = profits[0] + profits[1]
        pair_start = 0
        for i in range(len(profits) - 1):
            if profits[i] + profits[i + 1] > max_pair:
                max_pair = profits[i] + profits[i + 1]
                pair_start = i
        if max_pair > max_so_far:
            max_so_far = max_pair
            best_start = pair_start
            best_end = pair_start + 1

    return (max_so_far, best_start + 1, best_end + 1)


profits = input("输入利润: ").split()
profits = [float(x) for x in profits]
max_profit, buy, sell = max_profit(profits)
print(f"最大利润: {max_profit}, 买: {buy}, 卖: {sell}")


# ## 问题 5：
# 
# 观察下方的分治算法（divide-and-conquer algorithm）的伪代码, 回答下面问题
# 
# ```latex
# DoSomething(A,p,r)
# -----
# n := r-p+1
# if n=2 and A[p]>A[r] then
#     swap A[p] and A[r]
# else if n >= 3 then
#     m = ceil(2n/3)
#     DoSomething(A,p,p+m-1)
#     DoSomething(A,r-m+1,r)
#     DoSomething(A,p,p+m-1)  
#     
# ---
# first call: DoSomething(A,1,n)
# ```
# 
# note：$ceil(2n/3)=\left\lceil {2n/3} \right\rceil$；$:=$ 表示赋值, 等价于 $\to$；A是一个包含n的整数元素的数组, 
# 
# 1. 写出该算法时间复杂度的递归公式, 并求解其对应的渐进表示
# 2. 描述一下该算法的功能, 并判断是否是最高效的解决方案
# 3. 使用python编程实现上述算法或其对应的更高效版本的算法
# 

# idea：

# answer:

# In[ ]:


import math

def do_something(A, p, r):
    n = r - p + 1
    if n == 2 and A[p] > A[r]:
        A[p], A[r] = A[r], A[p]
    elif n >= 3:
        m = math.ceil(2 * n / 3)
        do_something(A, p, p + m - 1)
        do_something(A, r - m + 1, r)
        do_something(A, p, p + m - 1)

A = [5, 2, 8, 1, 9, 3]
do_something(A, 0, len(A)-1)
print("Sorted array:", A)
# your algorithm time complexity is :O(nlogn)


# ## 问题 6：
# 
# 给定一个大小为 `n` 的数组 `nums` , 返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。
# 
# 你可以假设数组是非空的, 并且给定的数组总是存在多数元素。
# 
# 1. 设计一个算法找到给定数组的多数元素, 分析算法设计思路, 计算算法时间复杂度, 使用python编程实现
# 2. \* 设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题, 分析算法设计思路, 使用python编程实现
# 
# e.g.:
# ---
# 1. nums=[3,2,3], 返回3
# 2. nums=[2,2,1,1,1,2,2], 返回2
# 

# idea：

# In[ ]:


# add your code here
def majorityElement(nums):
    count = 0
    candidate = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        if num == candidate:
            count += 1
        else:
            count -= 1
    
    return candidate
# 思路：找数组中出现次数大于 ⌊n/2⌋ 的多数元素。通过遍历数组维护候选元素和计数器（计数为 0 设新候选，相同加 1，不同减 1），最后候选元素为多数元素。
# your algorithm time complexity is :遍历一次，操作常数时间，O(n)，


# idea for 2\*：

# In[ ]:


# algorithm time complexity：O(n), space complexity:O(1)
# add your code here


# ## 问题 7：
# 
# 给定一个包含不同整数元素的数组 $ A[1..n]$ ,并且满足条件：$A[1]>A[2]$ 并且 $A[n-1]<A[n]$; 规定：如果一个元素比它两边的邻居元素都小, 即：$A[x]<A[x-1], A[x]<A[x+1]$ , 称这个元素A[x]为“局部最小”。通过遍历一次数组, 我们可以很容易在 $O(n)$的时间复杂度下找到一个局部最小值, 
# 
# 
# 1. 分析该问题, 设计一个算法在$O(logn)$的时间复杂度下找到一个局部最小(返回数值), 要求：分析算法设计思路, 并且使用python编程实现
# 2. \* 设计算法找出所有局部最小值, 分析算法设计思路, 并使用python编程实现
# 
# e.g.:
# ---
# A=[9, 3, 7, 2, 1, 4, 5 ] 时,  局部最小元素为 3, 1
# 

# idea：

# In[7]:


# add your code here
def find_local_minimum(A, left, right):
    if right - left == 1:
        if A[left] < A[right]:
            return A[left] 
        return A[right] 
    
    mid = left+((right-left)//2)
    
    if mid == 0 or mid == len(A) - 1:
        mid = (left + right + 1) // 2  # 避免越界
    
    if A[mid] < A[mid-1] and A[mid] < A[mid+1]:
        return A[mid]
    elif A[mid] > A[mid-1]:
        return find_local_minimum(A, left, mid-1)
    else:
        return find_local_minimum(A, mid+1, right)

# 测试
A = [9, 3, 7, 2, 1, 4, 5]
result = find_local_minimum(A, 0, len(A)-1)
print("Local minimum:", result)
# 思路：用二分查找找局部最小，比较中间点 mid 与两邻居，若 A[mid] 最小则返回，否则根据下降趋势递归左或右半
# your algorithm time complexity is : 每次二分规模减半，深度logn，时间复杂度为O(logn)


# idea:

# In[ ]:


# add your code here
# your algorithm time complexity is :


# ## 问题 8：
# 
# 给定包含n个不同数字的一组数, 寻找一种基于比较的算法在这组数中找到k个最小的数字, 并按顺序输出它们。
# 
# 1. 将n个数先进行排序, 然后按顺序输出最小的k个数。要求：选择合适的排序算法实现上述操作, 计算算法时间复杂度, 并使用python编程实现。
# 2. 建立一个包含这n个数的堆（heap）, 并且调用 k 次Extract-min 按顺序输出最小的k个数。使用往空堆中不断插入元素的方法建立堆, 分析这种方法建堆的时间复杂度, 并使用python编程实现
# 3. \* 假设数组中包含的数据总数目超过了计算机的存储能力, 请设计一个算法, 找到这堆数据的前k小的数值, 计算时间复杂度, 并使用python实现该算法, 假设计算机一定能存储k个数据。
# 
# e.g.：
# ---
# 数组arr=[5,4,3,2,6,1,88,33,22,107] 的前3个最小数据为：1, 2, 3
# 

# idea：

# In[ ]:


# add your code here
# 问题1：
def quicksort(arr, left, right):
    if left < right:
        pivot = arr[right]
        i = left - 1
        for j in range(left, right):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[right] = arr[right], arr[i + 1]
        partition = i + 1
        quicksort(arr, left, partition - 1)
        quicksort(arr, partition + 1, right)
def k_smallest_sorting(arr, k):
    n = len(arr)
    quicksort(arr, 0, n - 1)
    return arr[:k]
input_str = input("请输入一个数组：")
arr = [int(x) for x in input_str.split(',')]
k = int(input("输入查找最小元素个数 k："))
result = k_smallest_sorting(arr, k)
print("Smallest", k, "numbers (sorting):", result)
# 思路：用快速排序（平均 O(nlogn)）对数组排序，排序后取前𝑘个元素。快速排序适合通用场景，效率高。

#问题2：
import heapq
def k_smallest_heap(arr, k):
    heap = []
    for num in arr:
        heapq.heappush(heap, num)
    result = []
    for _ in range(k):
        result.append(heapq.heappop(heap))
    return result

input_str = input("请输入一个数组：")
arr = [int(x) for x in input_str.split(',')]
k = int(input("输入查找的最小元素个数 k："))
result = k_smallest_heap(arr, k)
print("Smallest", k, "numbers (heap):", result)
#思路：用最小堆存储数组元素。先从空堆开始，逐个插入元素，然后调用k次提取操作，依次输出最小的k个数。
# your algorithm time complexity is :
#问题1：快速排序：平均 O(nlogn)，最坏 O(𝑛2),总复杂度：O(nlogn)。
#问题2：推排序，插入n个元素到堆：O(nlogn)；提取k次最小值,每次为O(logn)，总共O(klogn)，因此总时间复杂度为O(nlogn+klogn)


# ## 问题 9：
# 
# **选择问题**:给定一个包含n个未排序值的数组A和一个$k≤n$的整数, 返回A中最小的第k项。
# 
# 在课堂上, 学了一个简单的O(n)随机算法来解决选择问题。事实上还有一种更复杂的最坏情况下时间复杂度为$O(n)$ 的选择算法。假设使用一个黑盒过程来实现这个O(n)选择算法: 给定一个数组A、 $p < r$ 和 k,  $BB(A, p, r, k)$ 可以在$O(r−p+ 1)$时间内找到并报告$A[p..r]$中第k小的项的下标。假设你可以在线性时间内处理Partition过程。
# 
# 1. 请分析如何修改 Quicksork 算法可以使其最差情况下的运行时间为 $O(nlogn)$, 使用伪代码实现, 并分析为何修改后的版本最差情况的运行时间为$O(nlogn)$
# 
# note: 伪代码中, 你可以直接调用用` BB(A,p,r,k)`这个函数用于表示在最坏情况下时间复杂度为$O(n)$的选择算法；
# 
# 
# 
# 2. 找到一个更好的算法报告数组A中的前k小的项, 使用伪代码表示你的算法, 并分析你算法的时间复杂度。
# 
# 举例：A=[13, 3, 7, 9, 11, 1, 15, 2, 8, 10, 12, 16, 14, 5], 当k=4时, 应该报告1, 2, 3, 4
# 
# note： 最直观的方法就是先将数组A排序, 然后从左向右报告其前k项, 这样操作的时间复杂度为$O(nlogn)$. 调用用` BB(A,p,r,k)`设计一个算法使其报告无序数组A的前k项, 满足时间复杂度好于$\Theta(nlogn)$, 并且当$k=\sqrt{n}$时, 你设计的算法时间复杂度应该为$\Theta(n)$.
# 
# 
# 
# 3. 给定一个大小为n的数组, 找到一个 时间复杂度为$O(n log k)$ 的算法, 该算法将A中的元素重新排序, 使它们被划分为k个部分, 每个部分的元素小于或等于下一部分的元素。假设n和k都是2的幂。使用伪代码表示你的算法, 并分析时间复杂度。
# 
# e.g.:
# ---
# 数组：[1,  3,  5,  7,  9,  11,  13,  15,  2,  4,  6,  8,  10,  12,  16,  14], k=4, 
# 
# 对应重新排序的数组为：[1,  3,  2,  4]  [7,  6,  5,  8]  [12,  11,  10,  9]  [13,  14,  16,  15]
# 
# 
# 

# idea：

# In[ ]:


# add your pseudo-code here


# ## 问题 10：
# 
# 给定一个包含m个**字符串**的数组A, 其中不同的字符串可能有不同的字符数, 但数组中所有字符串的字符总数为n。设计一个算法在 $O(n)$ 时间内对字符串进行排序, 分析算法设计方案, 计算其时间复杂度, 并基于python编程实现该算法。请注意, 假设字符串只包含"a","b",...,"z", 
# 
# 
# 
# 举例1：数组A=["a", "da", "bde", "ab", "bc", "abdc", "cdba"], 排序后的数组应该为：['a', 'ab', 'abdc', 'bc', 'bde', 'cdba', 'da']
# 
# 
# 
# 举例2：数组A=['ab', 'a', 'b', 'abc', 'ba', 'c'], 排序后的数组应该为：
# 
# ['a', 'ab', 'abc', 'b', 'ba', 'c']
# 
# 
# 
# 举例3：数组A=['aef', 'yzr', 'wr', 'ab', 'bhjc', 'lkabdc', 'pwcdba'],  排序后的数组应该为：['ab', 'aef', 'bhjc', 'lkabdc', 'pwcdba', 'wr', 'yzr']
# 
# 
# 
# note：
# 
# -  两个字符之间的比较可以考虑比较他们对应的ASCII码值；
# - python中可以使用`ord("a")`返回字符 “a”对应的ASCII值

# idea:

# In[8]:


# add your code here
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False

def sort_strings(A):
    root = TrieNode()
    for s in A:
        node = root
        for char in s:
            index = ord(char) - ord('a')
            if not node.children[index]:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.is_end = True
    result = []
    stack = []
    stack.append((root, ""))  
    
    while stack:
        node, current_prefix = stack.pop()
        if node.is_end and current_prefix:
            result.append(current_prefix)
        children = []
        for i in range(26):
            child = node.children[i]
            if child is not None:
                char = chr(ord('a') + i)
                children.append((child, current_prefix + char))
        for child in reversed(children):
            stack.append(child)
    
    return result
if __name__ == "__main__":
    A1 = ["a", "da", "bde", "ab", "bc", "abdc", "cdba"]
    print(sort_strings(A1)) 

# 思路：先将所有字符串插入字典树中，每个节点代表一个字符，路径表示字符串的字符序列，然后通过深度优先遍历字典树，按字典序（a到z）收集所有字符串，确保正确顺序。
# your algorithm time complexity is :
#插入字符串每个字符被处理一次，时间复杂度为 O(n)；遍历字典树，每个节点被访问一次，时间复杂度为 O(n)。


# In[ ]:




