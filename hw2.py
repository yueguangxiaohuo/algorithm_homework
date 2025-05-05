#!/usr/bin/env python
# coding: utf-8

# ## algorithm design and anlysis-2025 spring  homework 2
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
# > 给定一个已排序的链表的头 `head` ， *删除所有重复的元素，使每个元素只出现一次* 。返回 *已排序的链表* 。链表的类如下所示：
# 
# ```python
# class NodeList:
#     def __init__(self, val=None, right=None):
#         self.val   = val
#         self.right = right
# ```
# 
# 输入是一个数组，你首先需要将数组转化为链表，然后删除链表中的重复元素，再遍历链表元素，以一个数组的形式返回。请设计一个算法解决上述任务，分析算法设计思路，计算时间复杂度, 并基于python编程实现。
# 
# e.g.  输入：head=[1, 1, 2, 3, 3]   输出：[1, 2, 3]
# 
# ![image-20240502110020439](./fig/hw2q1.png)
# 
# 

# idea：

# In[1]:


# add your idea here
class NodeList:
    def __init__(self, val=None, right=None):
        self.val = val
        self.right = right
def delete_duplicates(head):
    if not head or not head.right:
        return head
    curr = head  # 当前节点
    while curr and curr.right:
        if curr.val == curr.right.val:
            curr.right = curr.right.right
        else:
            curr = curr.right
    return head
# 数组转链表
def array_to_list(arr):
    if not arr:  
        return None
    head = NodeList(arr[0])
    curr = head
    for num in arr[1:]:
        curr.right = NodeList(num)
        curr = curr.right
    return head
# 链表转数组
def list_to_array(head):
    arr = []
    curr = head
    while curr:
        arr.append(curr.val)
        curr = curr.right
    return arr
def process_list(arr):
    head = array_to_list(arr)
    head = delete_duplicates(head)
    result = list_to_array(head)
    return result
if __name__ == "__main__":
    test_arr = [1, 1, 2, 3, 3]
    print("输入:", test_arr)
    ans = process_list(test_arr)
    print("输出:", a，s)  
# 思路：先把数组转成链表，一个个节点连起来，因为链表是排序的，重复的肯定挨着，比较当前节点和下一个节点的值，相同就跳过下一个。最后把链表转回数组，遍历节点把值存下来。
# your algorithm time complexity is: 数组转链表时间复杂度为O(n)；去重也就是遍历一遍O(n)；链表转数组O(n)。总时间复杂度为O(n)。


# ## 问题 2  
# 
# > 下面是一个经典的算法问题：
# >
# > - 给定包含n个整数的一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的**数组下标**。假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。你可以按任意顺序返回答案。
# >
# > 由于要多次查找数组中元素的位置，为了提高查询效率可以使用哈希表来存储数组中的数据，在哈希表中查询一个元素的复杂度为O(1)。 已知python中的字典是使用哈希表实现的，即使用`dict[key]`查询对应的value时间复杂度为O(1), python提供了查询字典是否包含某个key的功能：`key in dict`，时间复杂度也是O(1)
# 
# 请根据上面信息，设计一个时间复杂度为O(n) 的算法，解决上述算法问题
# 
# e.g.   
# 
# 输入：nums=[2,7,11,15], target=9， 输出：[0，1]
# 
# 输入：nums=[3,2,4], target=6, 输出：[1,2]
# 
# 输入：nums=[3,3], target=6,  输出：[0,1]
# 

# In[2]:


# add your idea here
def two_sum(nums, target):
    num_dict = {}
    # 遍历数组
    for i in range(len(nums)):
        curr = nums[i]
        need = target - curr
        if need in num_dict:
            return [num_dict[need], i] 
        num_dict[curr] = i
if __name__ == "__main__":
    nums1 = [2, 7, 11, 15]
    target1 = 9
    print("输入:", nums1, "target:", target1)
    print("输出:", two_sum(nums1, target1)) 
# 思路：使用哈希表（dict）来存储数组元素和下标，然后遍历数组，对于当前元素 nums[i]，我们需要找 target - nums[i] 是否已经在哈希表中
# 如果找到了，说明之前某个元素加当前元素等于 target，返回它们的下标，如果没找到，把当前元素和它的下标存进哈希表，继续遍历。
# your algorithm time complexity is: 遍历数组一次，O(n)，哈希表查询和插入操作都是 O(1)，所以总时间复杂度为O(n)。


# ## 问题 3:   
# 
# > 栈是一种常用的数据结构，编译器中通常使用栈来实现表达式求值。
# >
# > 以表达式 $3+5 \times 8-6$​ 为例。编译器使用两个栈来完成运算，即一个栈保持操作数，另一个栈保存运算符。
# >
# > 1. 从左向右遍历表达式，遇到数字就压入操作数栈；
# >
# > 2. 遇到运算符，就与运算符栈的栈顶元素进行比较。如果比运算符栈顶元素的优先级高，就将当前运算符压入栈；如果比运算符栈顶元素的优先级低或者相同，从运算符栈中取栈顶运算符，从操作数栈的栈顶取 2 个操作数，然后进行计算，再把计算完的结果压入操作数栈，继续比较。
# >
# > 下图是 $3+5 \times 8-6$  这个表达式的计算过程：
# 
# ![figure](./fig/hw2q3.png)
# 
# 根据上述原理，请设计一个算法完成表达式的运算，当输入为表达式字符串，返回对应的计算结果。分析算法设计思路，计算时间复杂度，并基于python编程实现
# 
# **note：**
# 
# 1. 假设输入的表达式只会出现加（“+”），减 “-”， 乘“*”，除 “/” 四个运算符, 表达式中只会出现正整数
# 2. python中` str.isdigit()`函数可以判断字符串str是否为数字，
# 
# 
# 
# e.g. :
# ---
# 
# 1. 输入：“$3+5 * 8 -6$”   输出：37
# 
# 2. 输入：“$34+13*9 + 44-12/3$”  输出：191

# idea：

# In[3]:


# add your idea here
# 优先级函数
def priority(op):
    if op in ['+', '-']:
        return 1
    if op in ['*', '/']:
        return 2
    
def calculate(a, b, op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a // b  

def evaluate(expr):
    num_stack = []  
    op_stack = []   
    i = 0
    while i < len(expr):
        if expr[i] == ' ':
            i += 1
            continue
        # 如果是数字，提取完整数字
        if expr[i].isdigit():
            num = 0
            while i < len(expr) and expr[i].isdigit():
                num = num * 10 + int(expr[i])
                i += 1
            num_stack.append(num)
            continue
        
        # 如果是运算符
        if expr[i] in ['+', '-', '*', '/']:
            # 比较优先级
            while (op_stack and 
                   priority(op_stack[-1]) >= priority(expr[i])):
                # 弹出运算符和两个操作数计算
                op = op_stack.pop()
                b = num_stack.pop()
                a = num_stack.pop()
                result = calculate(a, b, op)
                num_stack.append(result)
            # 当前运算符入栈
            op_stack.append(expr[i])
        i += 1
    while op_stack:
        op = op_stack.pop()
        b = num_stack.pop()
        a = num_stack.pop()
        result = calculate(a, b, op)
        num_stack.append(result)
    return num_stack[0]
if __name__ == "__main__":
    expr1 = "3+5*8-6"
    print("表达式:", expr1)
    print("结果:", evaluate(expr1))

# 思路：遇到数字压入操作数栈，遇到运算符根据优先级（+、- 为 1，*、/ 为 2）决定直接压入运算符栈或弹出计算；
# 最后清空运算符栈计算剩余运算，返回操作数栈顶结果。
# your algorithm time complexity is: 遍历表达式为O(n)；运算符处理为O(n)。因此总时间复杂度为O(n)。


# ## 问题 4:  
# 
# > 星球碰撞问题：现有n个星球，在同一条直线上运行，如数组A所示，元素的绝对值表示星球的质量，负数表示星球自右向左运动，正数表示星球自左向右运动，当两个星球相撞的时候，质量小的会消失，大的保持不变，**质量相同的两个星球碰撞后自右向左运动的星球消失，自左向右的星球保持不变**，假设所有星球的速度大小相同。
# >
# > $ A=[23,-8, 9, -3, -7, 9, -23, 22] $
# 
# 请设计一个算法模拟星球的运行情况，输出最终的星球存续情况（输出一个数组），分析算法设计思路，计算时间复杂度，并基于python编程实现。
# 
# e.g.
# ---
# 1.  输入： A=[-3,-6,2,8, 5,-8,9,-2,1]， 输出：[-3, -6, 2, 8, 9, 1]
# 
# 2. 输入：A=[23,-8, 9, -3, -7, 9, -23, 22], 输出：[23, 22]
# 
# 

# ## 问题 5 
# 
# > 给定一个无序数组nums=[9,-3,-10,0,9,7,33]，请建立一个二叉搜索树存储数组中的所有元素，之后删除二叉树中的元素“0”，再使用中序遍历输出二叉搜索树中的所有元素。
# 
# 使用python编程完成上述任务，并计算时间复杂度
# 

# In[5]:


# add your idea here
def planet_collision(A):
    stack = []  
    for planet in A:
        while True:
            # 栈空或者没有碰撞
            if not stack:
                stack.append(planet)
                break
            top = stack[-1]
            # 判断是否会碰撞
            if top > 0 and planet < 0:
                top_mass = abs(top)
                curr_mass = abs(planet)
                if top_mass == curr_mass:
                    break 
                elif top_mass < curr_mass:
                    stack.pop() 
                    continue
                else:
                    break
            else:
                # 不会碰撞
                stack.append(planet)
                break
    
    return stack

if __name__ == "__main__":
    A1 = [-3, -6, 2, 8, 5, -8, 9, -2, 1]
    print("输入:", A1)
    print("输出:", planet_collision(A1))  
# 思路：使用栈从左到右遍历星球数组，栈顶星球向右且当前星球向左时发生碰撞，按质量大小和方向规则(质量小的消失，质量相等时右→左消失)
# 决定是否弹出栈顶或跳过当前星球，否则直接入栈。最终栈中星球即为存续结果。
# your algorithm time complexity is: 遍历数组一次，O(n);每颗星球最多入栈和出栈一次,栈操作 O(1);因此，总时间复杂度为O(n)


# ## 问题 6  
# 
# > 给定一个包含大写字母和小写字母的字符串 s ，返回 字符串包含的 **最长的回文子串的长度** 。请注意 区分大小写 。比如 "Aa" 不能当做一个回文字符串。
# >
# 
# 请设计一个算法解决上述问题，只需要输出最长回文子串的长度，分析算法设计思路，计算时间复杂度，并基于python编程实现
# 
# e.g. 输入： s="adccaccd"，  输出：7。 最长回文子串为："dccaccd", 长度为7
# 

# idea：

# In[6]:


# add your idea here
def longest_palindrome(s):
    if not s:
        return 0
    max_len = 1  
    n = len(s)
    
    # 中心扩展函数
    def expand_around_center(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    # 遍历每个中心
    for i in range(n):
        len1 = expand_around_center(i, i)
        len2 = expand_around_center(i, i + 1)
        max_len = max(max_len, len1, len2)
    
    return max_len

if __name__ == "__main__":
    s = "adccaccd"
    print("输入:", s)
    print("输出:", longest_palindrome(s))  

# 思路：使用中心扩展法遍历字符串每个字符作为回文中心，分别以单个字符（奇数长度）和相邻字符间（偶数长度）为中心向两边扩展，
# 比较左右字符（区分大小写）并记录最长回文长度。
# your algorithm time complexity is:遍历字符串时间复杂度为O(n)；每次中心扩展最多 O(n)，一共有2n个中心。因此总时间复杂度为O(n²)


# ## 问题 7 
# 
# > 沿一条长河流分散着n座房子。你可以把这条河想象成一条轴，房子是由它们在这条轴上的坐标按顺序排列的。你的公司想在河边的特定地点设置手机基站，这样每户人家都在距离基站4公里的范围内。输入可以看作为一个升序数组，数组元素的取值为大于等于0的正整数，你需要输出最小基站的数目，基站的位置。
# 
# 1. 给出一个时间复杂度为$O(n$) 的算法，使所使用的基站数量最小化，分析算法设计思路，使用python编程实现
# 2. 证明1.中算法产生了最优解决方案。
# 
# e.g. 
# 
# 输入： [1, 5, 12, 33, 34,35]  输出：基站数目为3， 基站位置为[1，12，33]
# 
# 

# idea：

# In[7]:


# add your idea here
def place_stations(houses):
    if not houses:
        return 0
    
    n = len(houses)
    count = 0  # 数量
    stations = []  # 位置
    i = 0  # 当前房子索引
    while i < n:
        count += 1
        station = houses[i]  
        stations.append(station)
        
        # 找到最后一个被当前基站覆盖的房子
        while i < n and houses[i] <= station + 4:
            i += 1
    
    return count, stations

if __name__ == "__main__":
    houses = [1, 5, 12, 33, 34, 35]
    print("输入:", houses)
    count, stations = place_stations(houses)
    print("基站数目:", count)
    print("基站位置:", stations)  
    
# 思路：使用贪心算法：从左到右遍历房子，尽可能少设置基站；
# 对于每个未覆盖的房子，放置一个基站，覆盖尽可能多的后续房子（坐标差 ≤ 4），然后跳到第一个未覆盖的房子继续。
# your algorithm time complexity is: 遍历字符串时间复杂度为O(n)；每次找覆盖范围内的房子是顺序遍历，整体 O(n)，因此总时间复杂度为O(n)。


# ## 问题 8  
# 
# > 给定由n个正整数组成的一个集合$S = \{a_1, a_2，···，a_n\}$和一个正整数W，设计一个算法确定是否存在S的一个子集 $K \subseteq S$, 使K中所有数之和为 $W$, 如果存在返回“True”，否则返回“False”
# 
# 请设计一个时间复杂度为$O(nW)$动态规划算法，解决上述问题，分析算法的设计思路，并且基于python编程实现（不需要输出子集）。
# 
# e.g. 
# 
# 输入：S = {1,4,7,3,5}， W = 11，输出：True。   因为K可以是{4,7}。
# 
# 

# idea：

# In[8]:


# add your idea here
def subset_sum(S, W):
    n = len(S)
    # dp[w] 表示能否用某些数凑出和为w
    dp = [False] * (W + 1)
    dp[0] = True  # 和为0
    
    for i in range(n):
        # 从大到小更新dp，避免重复使用同一个数
        for w in range(W, S[i] - 1, -1):
            dp[w] |= dp[w - S[i]]
    
    return dp[W]

if __name__ == "__main__":
    S = [1, 4, 7, 3, 5]
    W = 11
    print("输入: S =", S, "W =", W)
    print("输出:", subset_sum(S, W))  
# 思路：使用动态规划定义 dp[i][w] 表示前 i 个数能否组成和为 w，
# 通过状态转移，如不选第 i 个数（dp[i][w] = dp[i-1][w]）或选第 i 个数（当 w >= S[i-1]：dp[i][w] = dp[i-1][w - S[i-1]]）填充表格，
# 最终判断 dp[n][W] 是否为 True
# your algorithm time complexity is: 因为动态规划的表格大小为(n+1) × (W+1)，且填充每个格子时间复杂度为O(1)，因此总时间复杂度为O(nW)。


# 
# 
# ## 问题 9 
# 
# > 给定一个n个物品的集合。物体的重量为$w_1, w_2，…、w_n$，物品的价值分别是$v_1、v_2、…v_n$。给你**两个**重量为 $c$ 的背包。如果你带了一个东西，它可以放在一个背包里，也可以放在另一个背包里，但不能同时放在两个背包里。所有权重和价值都是正整数。
# 
# 1. 设计一个时间复杂度为 $O(nc^2)$ 的动态规划算法，确定可以放入两个背包的物体的最大价值。分析算法设计思路，并基于python编程实现
# 2. \* 修改1中的算法，输出每个背包的内容（物品对应下标）。
# 
# e.g.: 
# 
# 输入 V=[1,3,2,5,8,7], W=[1,3,2,5,8,7], c=7, 输出：最大价值=14，背包装的物品为：[6] [4，3] （同一个背包中物品装入顺序对结果无影响）  
# 

# idea：

# In[1]:


# add your idea here
def max_value(V, W, c):
    n = len(V)
    # dp[i][c1][c2] 表示前i个物品，背包1容量c1，背包2容量c2时的最大价值
    dp = [[[0] * (c + 1) for _ in range(c + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for c1 in range(c + 1):
            for c2 in range(c + 1):
                # 不选第i个物品
                dp[i][c1][c2] = dp[i-1][c1][c2]
                
                # 选第i个物品，放入第一个背包
                if c1 >= W[i-1]:
                    dp[i][c1][c2] = max(dp[i][c1][c2], dp[i-1][c1-W[i-1]][c2] + V[i-1])
                
                # 选第i个物品，放入第二个背包
                if c2 >= W[i-1]:
                    dp[i][c1][c2] = max(dp[i][c1][c2], dp[i-1][c1][c2-W[i-1]] + V[i-1])
    
    return dp[n][c][c]

if __name__ == "__main__":
    V = [1, 3, 2, 5, 8, 7]
    W = [1, 3, 2, 5, 8, 7]
    c = 7
    print("输入: V =", V, "W =", W, "c =", c)
    max_val = max_value(V, W, c)
    print("最大价值 =", max_val)  

# 思路：使用动态规划定义 dp[i][c1][c2] 表示前 i 个物品在两个背包容量 c1 和 c2 时的最大价值，
# 通过状态转移（不选第 i 个物品或选放入任一背包）取最大值，最终答案为 dp[n][c][c]。
# your algorithm time complexity is: 这里的动态规划表格大小为(n+1) × (c+1) × (c+1)，填充每个格子 O(1)，因此总的时间复杂度为O(nc^2)


# ## 问题 10 
# 
# > 给定两个字符串 $x[1..n]$ 和 $y[1..m]$，我们想通过以下操作将 $x$ 变换为 $y$ :
# >
# > **插入**：在 $x$ 中插入一个字符(在任何位置)；**删除**：从 $x$ 中删除一个字符(在任何位置)； **替换**：用另一个字符替换 $x$ 中的一个字符。
# >
# > 例如: $x = abcd$, $y = bcfe$，
# >
# > - 将 $x$ 转换为 $y$ 的一种可能方法是：1. 删除 $x$ 开头的 $a$, $x$变成 $bcd$； 2. 将 $x$ 中的字符 $d$ 替换为字符 $f$。$x$ 变成 $bcf$； 3. 在 $x$ 的末尾插入字符 $e$。$x$ 变成 $bcfe$。
# >
# > - 另一种可能的方法：1. 删除 $x$ 开头的 $a$,  $x$ 变成 $bcd$； 2. 在 $x$ 中字符 $d$ 之前插入字符 $f$。$x$ 变成 $bcfd$。3. 将 $x$ 中的字符 $d$ 替换为字符 $e$。$x$ 变成 $bcfe$。
# 
# 设计一个时间复杂度为 $O(mn)$ 的算法，返回将 $x$ 转换为 $y$ 所需的最少操作次数。分析算法设计思路，并基于python编程实现。
# 

# idea：

# In[2]:


# add your idea here
def min_edit_distance(x, y):
    n = len(x)
    m = len(y)
    
    # dp[i][j] 是把 x 前i个字符变成 y 前j个字符的最少操作数
    dp = []
    for i in range(n + 1):
        row = [0] * (m + 1)
        dp.append(row)
    
    # 初始空串转换
    for i in range(n + 1):
        dp[i][0] = i  
    for j in range(m + 1):
        dp[0][j] = j 
    
    # 填表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i-1] == y[j-1]: 
                dp[i][j] = dp[i-1][j-1]
            else:  
                insert = dp[i][j-1] + 1
                delete = dp[i-1][j] + 1
                replace = dp[i-1][j-1] + 1
                dp[i][j] = min(insert, delete, replace)
    
    return dp[n][m]

if __name__ == "__main__":
    x = "abcd"
    y = "bcfe"
    print("输入: x =", x, "y =", y)
    result = min_edit_distance(x, y)
    print("最少操作次数 =", result) 

# 思路：使用动态规划定义 dp[i][j] 为将 x[0:i] 转换为 y[0:j] 的最少操作次数，若 x[i-1] == y[j-1] 则无需操作，
# 否则通过插入、删除、替换三种操作取最小值更新 dp[i][j] 
# your algorithm time complexity is: 这里动态规划的表格大小是 (n+1) × (m+1)，填充每一个格个O(1)，因此总的时间复杂度为O(mn)


# In[ ]:




