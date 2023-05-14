
### 03. 数组中重复的数字

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        //
        int i = 0;
        int n = nums.length - 1;
        while (i <= n) {
            if (nums[i] == i) {
                i++;
                continue;
            }
            if (nums[nums[i]] == nums[i]) return nums[i];
            int tmp = nums[i];
            nums[i] = nums[tmp];
            nums[tmp] = tmp;
        }
        return -1;
    }
}
```

### 04. 二维数组中的查找

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        int i = matrix.length - 1;
        int j = 0;
        while (i >= 0 && j < matrix[0].length) {
            if (matrix[i][j] > target) {
                i--;
                continue;
            }
            if (matrix[i][j] < target) {
                j++;
                continue;
            }
            return true;
        }
        return false;
    }
}
```

### 05. 替换空格

```java
class Solution {
    public String replaceSpace(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (c == ' ') {
                stringBuilder.append("%20");
            } else {
                stringBuilder.append(c);
            }
        }
        return stringBuilder.toString();
    }
}
```

### 06. 从尾到头打印链表

```java
class Solution {
    private ArrayList<Integer> tmp = new ArrayList();

    public int[] reversePrint(ListNode head) {
        LinkedList<ListNode> stack = new LinkedList<>();
        while (head != null) {
            stack.addLast(head);
            head = head.next;
        }
        int[] ints = new int[stack.size()];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = stack.removeLast().val;
        }
        return ints;
    }

    void recur(ListNode node) {
        if (node == null) {
            return;
        }
        recur(node.next);
        tmp.add(node.val);
    }

}
```

### 07. 重建二叉树

```java
class Solution {
    HashMap<Integer, Integer> hashMap = new HashMap<>();
    int[] preorder;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        for (int i = 0; i < inorder.length; i++) {
            hashMap.put(inorder[i], i);
        }
        return recur(0, 0, preorder.length - 1);
    }

    private TreeNode recur(int rooti, int lefti, int righti) {
        if (lefti > righti) return null;
        int middlei = hashMap.get(preorder[rooti]);
        TreeNode treeNode = new TreeNode(preorder[rooti]);
        treeNode.left = recur(rooti + 1, lefti, middlei - 1);
        treeNode.right = recur(rooti + middlei - lefti + 1, middlei + 1, righti);

        return treeNode;
    }

}
```

### 09. 用两个栈实现队列

```java
    class CQueue {
    LinkedList<Integer> stackA, stackB;

    public CQueue() {
        stackB = new LinkedList<>();
        stackA = new LinkedList<>();

    }

    public void appendTail(int value) {
        stackA.addLast(value);
    }

    public int deleteHead() {
        if (!stackB.isEmpty()) return stackB.removeLast();
        if (stackA.isEmpty()) return -1;
        while (!stackA.isEmpty()) {
            stackB.addLast(stackA.removeLast());
        }
        return stackB.removeLast();
    }
}
```

### 10- I. 斐波那契数列

```java
class Solution {
    int[] arr;
    int mod = 1000000007;

    public int fib(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        if (n == 2) return 1;
        arr = new int[n + 1];
        arr[1] = 1;
        arr[2] = 1;
        for (int i = 3; i < arr.length; i++) {
            arr[i] = (arr[i - 1] + arr[i - 2]) % mod;
        }
        return arr[n];


    }
}
```

### 10- II. 青蛙跳台阶问题

```java
class Solution {
    int mod = 1000000007;

    public int numWays(int n) {
        if (n == 0) return 1;
        if (n == 1) return 1;
        if (n == 2) return 2;
        int[] nums = new int[n + 1];
        nums[0] = 0;
        nums[1] = 1;
        nums[2] = 2;
        for (int i = 3; i < nums.length; i++) {
            nums[i] = (nums[i - 1] + nums[i - 2]) % mod;
        }
        return nums[n];
    }
}
```

### 11. 旋转数组的最小数字

```java
class Solution {
    public int minArray(int[] numbers) {
        int l = 0;
        int r = numbers.length - 1;
        int m;
        while (r - l > 0) {
            m = (r + l) / 2;
            if (numbers[r] < numbers[m]) l = m + 1;
            else if (numbers[m] < numbers[r]) r = m;
            else r--;
        }
        return numbers[r];
    }
}
```

### 12. 矩阵中的路径

```java
class Solution {
    char[] chars;

    public boolean exist(char[][] board, String word) {
        chars = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(board, word, i, j, 0)) return true;
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, String word, int i, int j, int k) {
        if (i >= board.length || i < 0 || j >= board[0].length || j < 0 || board[i][j] != chars[k]) {
            return false;
        }
        if (k == word.length() - 1) return true;
        board[i][j] = '\0';
        boolean res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i - 1, j, k + 1) || dfs(board, word, i, j - 1, k + 1);
        board[i][j] = chars[k];
        return res;
    }
}
```

### 13. 机器人的运动范围

```java
    class Solution {
    int m, n, k;
    boolean[][] visited;

    public int movingCount(int m, int n, int k) {
        this.m = m;
        this.n = n;
        this.k = k;
        visited = new boolean[m][n];
        return dfs(0, 0, 0, 0);
    }

    public int dfs(int i, int j, int si, int sj) {
        if (i >= m || j >= n || si + sj > k || visited[i][j]) {
            return 0;
        }
        visited[i][j] = true;
        return 1 + dfs(i + 1, j, (i + 1) % 10 == 0 ? si - 8 : si + 1, sj) + dfs(i, j + 1, si, (j + 1) % 10 == 0 ? sj - 8 : sj + 1);
    }
}
```

### 14- I. 剪绳子

`给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。`

```java
    class Solution {
    public int cuttingRope(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 1;
        for (int i = 2; i < dp.length; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(dp[i], Math.max((i - j) * j, dp[i - j] * j));
            }
        }
        return dp[n];
    }
}
```

### 14- II. 剪绳子 II

` 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m - 1] 。请问 k[0]*k[1]*...*k[m - 1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1`

```java
    class Solution {
    public int cuttingRope(int n) {
        if (n == 2) return 1;
        if (n == 3) return 2;
        if (n == 4) return 4;

        long ans = 1;
        int mod = 1000000007;
        while (n > 4) {
            n -= 3;
            ans = ans * 3 % mod;
        }
        ans = ans * n % mod;
        return (int) ans;
    }
}
```

### 15. 二进制中1的个数

```java
    public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {

        int res = 0;
        while (n != 0) {
            res++;
            n &= n - 1;
        }
        return res;
    }
}
```

### 16. 数值的整数次方

```java
    class Solution {
    public double myPow(double x, int n) {
        if (x == 0) return 0;
        long b = n;
        double res = 1.0;
        if (b < 0) {
            x = 1 / x;
            b = -b;
        }
        while (b > 0) {
            if ((b & 1) == 1) res = res * x;
            x = x * x;
            b = b >> 1;
        }
        return res;
    }
}
```

### 17. 打印从1到最大的n位数

```java
    class Solution {
    int[] nums;
    int count = 0;

    public int[] printNumbers(int n) {
        nums = new int[(int) Math.pow(10, n) - 1];//warning! need -1

        for (int digit = 1; digit < n + 1; digit++) {
            for (char j = '0'; j <= '9'; j++) {
                char[] chars = new char[digit];
                chars[0] = j;
                dfs(1, digit, chars);
            }

        }
        return nums;
    }

    public void dfs(int index, int digit, char[] num) {
        if (index == digit) {
            nums[count++] = Integer.parseInt(new String(num));
            return;
        }
        for (char c = '0'; c <= '9'; c++) {
            num[index] = c;
            dfs(index + 1, digit, num);
        }
    }


}
```

### 18. 删除链表的节点

```java
    /**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode(int x) { val = x; }
 * }
 */

public class Solution {
    public static ListNode deleteNode(ListNode head, int val) {
        ListNode pre = null;
        ListNode cur = head;
        ListNode next = head.next;
        while (cur != null) {
            if (cur.val == val) {
                if (pre == null) {
                    return next;
                } else {
                    pre.next = next;
                    return head;
                }
            }
            pre = cur;
            next = cur.next == null ? null : cur.next.next;
            cur = cur.next;
        }

        return null;
    }
}
```

### 19. 正则表达式匹配

```java
    class Solution {
    String s;
    String p;

    public boolean isMatch(String s, String p) {
        this.s = s;
        this.p = p;
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[0][0] = true;
        for (int i = 0; i < s.length() + 1; i++) {
            for (int j = 1; j < p.length() + 1; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2];
                    if (isEqual(i, j - 1)) {
                        dp[i][j] = dp[i][j] || dp[i - 1][j];
                    }
                } else {
                    if (isEqual(i, j)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else {
                        dp[i][j] = false;
                    }

                }

            }
        }
        return dp[s.length()][p.length()];
    }


    public boolean isEqual(int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }


}
```

### 20. 表示数值的字符串

```java

//-.4，0.4 = .4；2.、3. = 2、3，小数点前有数，后面可以不跟数代表原数
//注意e8即10的8次幂（8次方），也可以是e-7，但题目要求必须跟整数
//题目规定是数值前后可有空格，中间不能有，这个情况要考虑清楚。s：符号、d：数字
class Solution {
    public boolean isNumber(String s) {
        Map[] states = {
                //0：规定0是初值，字符串表示数值，有4种起始状态，开头空格、符号、数字、前面没有数的小数点
                //其中 开头空格 还是指向states[0]，上一位是 开头空格，下一位可以是 空格、符号、数字、前面没有数的小数点
                new HashMap<>() {{
                    put(' ', 0);
                    put('s', 1);
                    put('d', 2);
                    put('.', 4);
                }},
                //1：上一位是符号，符号位后面可以是 数字、前面没有数的小数点
                new HashMap<>() {{
                    put('d', 2);
                    put('.', 4);
                }},
                //2：上一位是数字，数字的下一位可以是 数字、前面有数的小数点、e、结尾空格
                new HashMap<>() {{
                    put('d', 2);
                    put('.', 3);
                    put('e', 5);
                    put(' ', 8);
                }},
                //3：上一位是前面有数的小数点，下一位可以是 数字、e（8.e2 = 8e2，和2的情况一样）、结尾空格
                new HashMap<>() {{
                    put('d', 3);
                    put('e', 5);
                    put(' ', 8);
                }},
                //4：上一位是前面没有数的小数点，下一位只能是 数字（符号肯定不行，e得前面有数才行）              
                new HashMap<>() {{
                    put('d', 3);
                }},
                //5：上一位是e，下一位可以是 符号、数字
                new HashMap<>() {{
                    put('s', 6);
                    put('d', 7);
                }},
                //6：：上一位是e后面的符号，下一位只能是 数字
                new HashMap<>() {{
                    put('d', 7);
                }},
                //7：上一位是e后面的数字，下一位可以是 数字、结尾空格
                new HashMap<>() {{
                    put('d', 7);
                    put(' ', 8);
                }},
                //8：上一位是结尾空格，下一位只能是 结尾空格
                new HashMap<>() {{
                    put(' ', 8);
                }}
        };
        int p = 0;
        char t;
        //遍历字符串，每个字符匹配对应属性并用t标记，非法字符标记？
        for (char c : s.toCharArray()) {
            if (c >= '0' && c <= '9') t = 'd';
            else if (c == '+' || c == '-') t = 's';
            else if (c == 'e' || c == 'E') t = 'e';
            else if (c == '.' || c == ' ') t = c;
            else t = '?';
            //当前字符标记和任何一种当前规定格式都不匹配，直接返回false
            if (!states[p].containsKey(t)) return false;
            //更新当前字符的规定格式，进入下一个规定的Map数组
            p = (int) states[p].get(t);
        }
        //2（正、负整数）、3（正、负小数）、7（科学计数法）、8（前三种形式的结尾加上空格）
        //只有这四种才是正确的结尾
        return p == 2 || p == 3 || p == 7 || p == 8;
    }
}
```

### 21. 调整数组顺序使奇数位于偶数前面

```java
    class Solution {
    public int[] exchange(int[] nums) {
        int i = 0;
        int j = nums.length - 1;
        int tmp = 0;

        while (i < j) {
            while (i < nums.length && (nums[i] & 1) == 1) i++;
            while (j >= 0 && (nums[j] & 1) == 0) j--;
            tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }

        return nums;
    }
}
```

### 22. 链表中倒数第k个节点

```java
    /**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode cur = head;
        ListNode next = cur.next;
        while (k > 1 && next != null) {
            next = next.next;
            k--;
        }
        while (next != null) {
            cur = cur.next;
            next = next.next;
        }
        return cur;
    }
}
```

### 24. 反转链表

```java
    /**
 * Definition for singly-linked list.
 * public class ListNode {
 * int val;
 * ListNode next;
 * ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode pre = null;
        ListNode next = head.next;
        while (next != null) {
            cur.next = pre;
            pre = cur;
            cur = next;
            next = next.next;
        }
        return pre;
    }
}

```

### 25. 合并两个排序的链表

```java
    class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;//warning!!

        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;//warning!!
        }
    }
}
```

### 26. 树的子结构

```java
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        return ((A != null) && (B != null)) && (isEqual(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B));
    }

    public boolean isEqual(TreeNode A, TreeNode B) {
        if (B == null) return true;
        if (A == null || A.val != B.val) return false;
        return isEqual(A.right, B.right) && isEqual(A.left, B.left);
    }

}
```

### 27. 二叉树的镜像

```java
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if (root == null) return null;
        TreeNode tmp = root.right;
        root.right = mirrorTree(root.left);
        root.left = mirrorTree(tmp);
        return root;
    }
}
```

### 28. 对称的二叉树

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return root == null || recur(root.left, root.right);
    }

    public boolean recur(TreeNode left, TreeNode right) {
        if (left == null & right == null) return true;
        if (left == null || right == null || right.val != left.val) return false;
        return recur(left.left, right.right) && recur(left.right, right.left);
    }
}
```

### 29. 顺时针打印矩阵

```java
    class Solution {
    public int[] spiralOrder(int[][] matrix) {
        if(matrix.length == 0) return new int[0];
        int l = 0, r = matrix[0].length - 1, t = 0, b = matrix.length - 1, x = 0;
        int[] res = new int[(r + 1) * (b + 1)];
        while(true) {
            for(int i = l; i <= r; i++) res[x++] = matrix[t][i]; // left to right.
            if(++t > b) break;
            for(int i = t; i <= b; i++) res[x++] = matrix[i][r]; // top to bottom.
            if(l > --r) break;
            for(int i = r; i >= l; i--) res[x++] = matrix[b][i]; // right to left.
            if(t > --b) break;
            for(int i = b; i >= t; i--) res[x++] = matrix[i][l]; // bottom to top.
            if(++l > r) break;
        }
        return res;
    }
}
```

### 30. 包含min函数的栈

```java
    class MinStack {
    Stack<Integer> arrl;
    Stack<Integer> minArray;

    public MinStack() {
        this.arrl = new Stack<>();

        this.minArray = new Stack<>();

    }

    public void push(int x) {
        if (minArray.empty() || minArray.peek() >= x) {// warning
            minArray.push(x);
        }
        arrl.push(x);
    }

    public void pop() {
        if (arrl.pop().equals(minArray.peek())) {
            minArray.pop();
        }
    }

    public int top() {
        return arrl.peek();
    }

    public int min() {
        return minArray.peek();
    }
}

```

### 31. 栈的压入、弹出序列

```java
    class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();

        int i = 0;
        for (int num : pushed) {
            stack.push(num);
            while (!stack.isEmpty() && stack.peek() == popped[i]) {
                stack.pop();
                i++;
            }
        }


        return stack.isEmpty();
    }
}

```

### 32 - I. 从上到下打印二叉树

` 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。`

```java
    class Solution {
    public int[] levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>() {{
            add(root);
        }};
        ArrayList<Integer> arr = new ArrayList<>();
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node != null) {
                arr.add(node.val);
                queue.offer(node.left);
                queue.offer(node.right);
            }
        }
        int[] nums = new int[arr.size()];
        for (int i = 0; i < arr.size(); i++) {
            nums[i] = arr.get(i);
        }

        return nums;
    }
}

```

### 32 - II. 从上到下打印二叉树 II

```java
    class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) return new LinkedList<>() {{
            add(new LinkedList<>());
        }};
        Queue<TreeNode> queue = new LinkedList<>() {{
            offer(root);
        }};
        List<List<Integer>> list = new LinkedList<>();

        while (!queue.isEmpty()) {
            LinkedList<Integer> lst = new LinkedList<>();
            for (int i = queue.size(); i > 0; i--) {//warning
                TreeNode node = queue.poll();
                lst.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            list.add(lst);
        }
        return list;
    }
}

```

### 32 - III. 从上到下打印二叉树 III

` 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。 `

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        if (root != null) queue.offer(root);
        List<List<Integer>> list = new LinkedList<>();
        while (!queue.isEmpty()) {
            LinkedList<Integer> integers = new LinkedList<>();
            for (int i = queue.size(); i > 0; i--) {//warning
                TreeNode node = queue.poll();
                if (list.size() % 2 == 0) integers.addLast(node.val);
                else integers.addFirst(node.val);

                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            list.add(integers);
        }
        return list;
    }
}

```

### 33. 二叉搜索树的后序遍历序列

`输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。`

```java
    class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return recur(postorder, 0, postorder.length - 1);
    }


    public boolean recur(int[] postOrder, int i, int j) {
        if (i > j) return true;//warning
        int p = i;
        while (postOrder[p] < postOrder[j]) p++;
        int middlei = p;
        while (postOrder[p] > postOrder[j]) p++;
        return p == j && recur(postOrder, i, middlei - 1) && recur(postOrder, middlei, j - 1);
    }
}
```

### 34. 二叉树中和为某一值的路径

```java
    class Solution {
    List<List<Integer>> list = new LinkedList<>();
    LinkedList<Integer> path = new LinkedList<>();

    public List<List<Integer>> pathSum(TreeNode root, int target) {
        recur(root, target);
        return list;
    }

    public void recur(TreeNode treeNode, int target) {
        if (treeNode == null) return;
        int val = treeNode.val;
        target -= val;
        path.add(val);
        if ((target == 0) && (treeNode.left == null) && (treeNode.right == null)) {
            list.add(new LinkedList<>(path));//warning
        }
        recur(treeNode.left, target);
        recur(treeNode.right, target);
        path.removeLast();
    }
}

```

### 35. 复杂链表的复制

`请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。`

```java
    class Solution {
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Node cur = head;
        HashMap<Node, Node> nodeNodeHashMap = new HashMap<>();
        while (cur != null) {
            nodeNodeHashMap.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        while (cur != null) {
            nodeNodeHashMap.get(cur).next = nodeNodeHashMap.get(cur.next);
            nodeNodeHashMap.get(cur).random = nodeNodeHashMap.get(cur.random);
            cur = cur.next;//warning
        }
        return nodeNodeHashMap.get(head);
    }
}
```

### 36. 二叉搜索树与双向链表

```java
class Solution {
    Node pre, head;

    public Node treeToDoublyList(Node root) {
        if (root == null) return null;
        recur(root);
        pre.right = head;
        head.left = pre;
        return head;
    }

    private void recur(Node root) {
        if (root == null) return;
        recur(root.left);
        if (pre == null) {
            head = root;
        } else {
            pre.right = root;
        }
        root.left = pre;
        pre = root;
        recur(root.right);
    }
}
```

### 37. 序列化二叉树

```java
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "[]";
        Queue<TreeNode> queue = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node != null) {
                sb.append(Integer.toString(node.val) + ",");
                queue.offer(node.left);
                queue.offer(node.right);
            } else {
                sb.append("null,");
            }
        }
        sb.deleteCharAt(sb.length() - 1);
        sb.append("]");
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.equals("[]")) return null;
        String[] strings = data.substring(1, data.length() - 1).split(",");
        int i = 1;
        TreeNode root = new TreeNode(Integer.parseInt(strings[0]));
        Queue<TreeNode> queue = new LinkedList<>() {{
            add(root);
        }};
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (!strings[i].equals("null")) {
                node.left = new TreeNode(Integer.parseInt(strings[i]));
                queue.offer(node.left);
            }
            i++;
            if (!strings[i].equals("null")) {
                node.right = new TreeNode(Integer.parseInt(strings[i]));
                queue.offer(node.right);
            }
            i++;
        }
        return root;

    }
}

```

### 38. 字符串的排列

`输入一个字符串，打印出该字符串中字符的所有排列。`

```java
class Solution {
    String str;
    char[] chars;
    ArrayList<String> arrayList = new ArrayList<>();

    public String[] permutation(String s) {
        this.str = s;
        chars = s.toCharArray();
        dfs(0);
        return arrayList.toArray(new String[arrayList.size()]);//warning
    }

    public void dfs(int x) {
        if (x == str.length() - 1) {
            arrayList.add(new String(chars));
        }

        HashSet<Character> set = new HashSet<>();
        for (int i = x; i < str.length(); i++) {
            if (set.contains(chars[i])) {
                continue;
            }
            set.add(chars[i]);
            swap(x, i);
            dfs(x + 1);
            swap(x, i);

        }

    }

    public void swap(int i, int j) {
        char tmp = chars[i];
        chars[i] = chars[j];
        chars[j] = tmp;
    }
}
```

### 39. 数组中出现次数超过一半的数字

```java
class Solution {
    public int majorityElement(int[] nums) {
        int x = 0;
        int vote = 0;

        for (int i = 0; i < nums.length; i++) {
            if (vote == 0) x = nums[i];
            vote += (x == nums[i] ? 1 : -1);
        }

        return x;
    }
}
```

### 40. 最小的k个数

```java
    class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        quickSort(arr, 0, arr.length - 1);
        // [0, 1, 2, 1, 4]
        return Arrays.copyOf(arr, k);
    }

    private void quickSort(int[] arr, int l, int r) {
        if (l >= r) return;
        int i = l;
        int j = r;
        while (i < j) {
            while (i < j && arr[j] >= arr[l]) j--;//warning
            while (i < j && arr[i] <= arr[l]) i++;
            swap(arr, i, j);
        }
        swap(arr, l, i);
        quickSort(arr, l, i - 1);
        quickSort(arr, i + 1, r);
    }

    private void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

```

### 41. 数据流中的中位数

`如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。`

```java
class MedianFinder {
    Queue<Integer> A;
    Queue<Integer> B;

    public MedianFinder() {
        A = new PriorityQueue<>();// 小顶堆  保存较大一半
        B = new PriorityQueue<>((x, y) -> (y - x));// 大顶堆  保存较小一半
    }

    public void addNum(int num) {
        if (A.size() != B.size()) {
            A.add(num);
            B.add(A.poll());
        } else {
            B.add(num);
            A.add(B.poll());
        }
    }

    public double findMedian() {
        return A.size() != B.size() ? A.peek() : (A.peek() + B.peek()) / 2.0;
    }
}

```

### 42. 连续子数组的最大和

```java
    class Solution {
    public int maxSubArray(int[] nums) {
        int[] maxs = new int[nums.length];
        int res = nums[0];
        maxs[0] = res;
        for (int i = 1; i < nums.length; i++) {
            maxs[i] = nums[i] + Math.max(maxs[i - 1], 0);// warning
            res = Math.max(res, maxs[i]);
        }
        return res;
    }
}
```

### 43. 1～n 整数中 1 出现的次数

```java
    class Solution {
    public int countDigitOne(int n) {
        int res = 0;
        int cur = n % 10;
        int low = 0;
        int hight = n / 10;
        int digit = 1;
        while (hight != 0 || cur != 0) {
            if (cur == 1) res += hight * digit + low + 1;
            else if (cur == 0) res += hight * digit;
            else res += (hight + 1) * digit;
            low += cur * digit;
            cur = hight % 10;
            hight /= 10;
            digit *= 10;
        }
        return res;

    }
}

```

### 44. 数字序列中某一位的数字

```java
    class Solution {
    public int findNthDigit(int n) {
        long start = 1;
        long count = 9;
        int digit = 1;

        while (count < n) {
            n -= count;
            digit = digit + 1;
            start *= 10;
            count = 9 * digit * start;
        }

        long num = start + (n - 1) / digit;

        return Long.toString(num).charAt((n - 1) % digit) - '0';
    }
}
```

### 45. 把数组排成最小的数

```java
    class Solution {
    public String minNumber(int[] nums) {
        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strings[i] = String.valueOf(nums[i]);
        }
        quickSort(strings, 0, strings.length - 1);

        StringBuilder sb = new StringBuilder();
        for (String str : strings) {
            sb.append(str);
        }
        return sb.toString();
    }

    private void quickSort(String[] strings, int i, int j) {
        if (i >= j) return;
        int l = i;
        int r = j;
        String tmp = strings[i];
        while (i < j) {
            while (i < j && (strings[j] + strings[l]).compareTo((strings[l] + strings[j])) >= 0) j--;
            while (i < j && (strings[i] + strings[l]).compareTo((strings[l] + strings[i])) <= 0) i++;
            String tmp2 = strings[i];
            strings[i] = strings[j];
            strings[j] = tmp2;
        }
        strings[l] = strings[i];
        strings[i] = tmp;
        quickSort(strings, l, i - 1);
        quickSort(strings, i + 1, r);
    }
}
```

### 46. 把数字翻译成字符串

```java
    class Solution {
    public int translateNum(int num) {

        String tmp = "";
        int[] dp = new int[String.valueOf(num).length() + 1];
        String numString = String.valueOf(num);

        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i < dp.length; i++) {
            tmp = numString.substring(i - 2, i);
            if (tmp.compareTo("10") >= 0 && tmp.compareTo("25") <= 0) {
                dp[i] = dp[i - 2] + 2;
            } else {
                dp[i] = dp[i - 1];
            }
        }

        return 0;

    }
}

```

### 47. 礼物的最大价值

```java
    class Solution {
    public int maxValue(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        dp[0][0] = grid[0][0];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (i == 0 && j == 0) continue;
                if (j == 0) dp[i][j] = grid[i][j] + dp[i - 1][j];
                else if (i == 0) dp[i][j] = grid[i][j] + dp[i][j - 1];
                else dp[i][j] = grid[i][j] + Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return dp[grid.length - 1][grid[0].length - 1];
    }
}
```

### 48. 最长不含重复字符的子字符串

```java
    class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.equals("")) return 0;
        HashMap<Character, Integer> map = new HashMap<>();
        char[] chars = s.toCharArray();
        int i = 0;
        int res = 1;
        for (int j = 0; j < chars.length; j++) {
            Character val = chars[j];
            if (map.containsKey(val)) {
                i = Math.max(map.get(val) + 1, i);//warning
            }
            map.put(val, j);
            res = Math.max(res, j - i + 1);
        }
        return res;
    }
}
```

### 49. 丑数

`我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。`

```java
    class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.equals("")) return 0;
        HashMap<Character, Integer> map = new HashMap<>();
        char[] chars = s.toCharArray();
        int i = 0;
        int res = 1;
        for (int j = 0; j < chars.length; j++) {
            Character val = chars[j];
            if (map.containsKey(val)) {
                i = Math.max(map.get(val) + 1, i);//warning
            }
            map.put(val, j);
            res = Math.max(res, j - i + 1);
        }
        return res;
    }
}
```

### 50. 第一个只出现一次的字符

```java
    class Solution {
    public char firstUniqChar(String s) {
        HashMap<Character, Boolean> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, !map.containsKey(c));
        }
        for (char c : s.toCharArray()) {
            if (map.get(c)) {
                return c;
            }
        }
        return ' ';
    }
}
```

### 51. 数组中的逆序对

```java
    class Solution {
    int[] tmp, numbers;
    int res = 0;

    public int reversePairs(int[] nums) {
        this.numbers = nums;
        tmp = new int[nums.length];
        mergeSort(nums, 0, nums.length - 1);
        return res;
    }

    public void mergeSort(int[] nums, int l, int r) {
        if (l >= r) return;

        int m = (r + l) / 2;
        int i = l;
        int j = m + 1;
        mergeSort(nums, l, m);
        mergeSort(nums, m + 1, r);

        for (int k = l; k < r + 1; k++) {
            tmp[k] = nums[k];
        }

        for (int k = l; k < r + 1; k++) {
            if (i == m + 1) {
                nums[k] = tmp[j++];
            } else if ((j == r + 1) || tmp[i] <= tmp[j]) {
                nums[k] = tmp[i++];
            } else {
                nums[k] = tmp[j++];
                res += m - i + 1;
            }
        }
    }
}
```

### 52. 两个链表的第一个公共节点

```java
    class Solution {
    ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA;
        ListNode B = headB;
        while (A != B) {
            A = A == null ? headB : A.next;
            B = B == null ? headA : B.next;
        }
        return A;
    }
}
```

### 53 - I. 在排序数组中查找数字 I
`统计一个数字在排序数组中出现的次数。`
```java
class Solution {
    public int search(int[] nums, int target) {
        return helper(nums, target) - helper(nums, target - 1);
    }
    int helper(int[] nums, int tar) {
        int i = 0, j = nums.length - 1;
        while(i <= j) {
            int m = (i + j) / 2;
            if(nums[m] <= tar) i = m + 1;
            else j = m - 1;
        }
        return i;
    }
}
```

### 53 - II. 0～n-1中缺失的数字

```java
class Solution {
    public int missingNumber(int[] nums) {
        int i = 0, j = nums.length - 1;
        while(i <= j) {
            int m = (i + j) / 2;
            if(nums[m] == m) i = m + 1;
            else j = m - 1;
        }
        return i;
    }
}
```

### 54. 二叉搜索树的第k大节点

```java
class Solution {
    int res, k;
    public int kthLargest(TreeNode root, int k) {
        this.k = k;
        dfs(root);
        return res;
    }
    void dfs(TreeNode root) {
        if(root == null) return;
        dfs(root.right);
        if(k == 0) return;
        if(--k == 0) res = root.val;
        dfs(root.left);
    }
}
```

### 55 - I. 二叉树的深度

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
}
```

### 55 - II. 平衡二叉树

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return recur(root) != -1;
    }

    private int recur(TreeNode root) {
        if (root == null) return 0;
        int left = recur(root.left);
        if(left == -1) return -1;
        int right = recur(root.right);
        if(right == -1) return -1;
        return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
    }
}
```

### 56 - I. 数组中数字出现的次数
`一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。`
```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        int x = 0, y = 0, n = 0, m = 1;
        for(int num : nums)               // 1. 遍历异或
            n ^= num;
        while((n & m) == 0)               // 2. 循环左移，计算 m
            m <<= 1;
        for(int num: nums) {              // 3. 遍历 nums 分组
            if((num & m) != 0) x ^= num;  // 4. 当 num & m != 0
            else y ^= num;                // 4. 当 num & m == 0
        }
        return new int[] {x, y};          // 5. 返回出现一次的数字
    }
}
```

### 56 - II. 数组中数字出现的次数 II
`在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。`
```java
class Solution {
    public int singleNumber(int[] nums) {
        int[] counts = new int[32];
        for(int num : nums) {
            for(int j = 0; j < 32; j++) {
                counts[j] += num & 1;
                num >>>= 1;
            }
        }
        int res = 0, m = 3;
        for(int i = 0; i < 32; i++) {
            res <<= 1;
            res |= counts[31 - i] % m;
        }
        return res;
    }
}
```

### 57. 和为s的两个数字
`输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。`
```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int i = 0, j = nums.length - 1;
        while(i < j) {
            int s = nums[i] + nums[j];
            if(s < target) i++;
            else if(s > target) j--;
            else return new int[] { nums[i], nums[j] };
        }
        return new int[0];
    }
}
```

### 57 - II. 和为s的连续正数序列
`输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）`
```java
class Solution {
    public int[][] findContinuousSequence(int target) {
        int i = 1, j = 2, s = 3;
        List<int[]> res = new ArrayList<>();
        while(i < j) {
            if(s == target) {
                int[] ans = new int[j - i + 1];
                for(int k = i; k <= j; k++)
                    ans[k - i] = k;
                res.add(ans);
            }
            if(s >= target) {
                s -= i;
                i++;
            } else {
                j++;
                s += j;
            }
        }
        return res.toArray(new int[0][]);
    }
}
```

### 58 - I. 翻转单词顺序

```java
class Solution {
    public String reverseWords(String s) {
        String[] strs = s.trim().split(" "); // 删除首尾空格，分割字符串
        StringBuilder res = new StringBuilder();
        for(int i = strs.length - 1; i >= 0; i--) { // 倒序遍历单词列表
            if(strs[i].equals("")) continue; // 遇到空单词则跳过
            res.append(strs[i] + " "); // 将单词拼接至 StringBuilder
        }
        return res.toString().trim(); // 转化为字符串，删除尾部空格，并返回
    }
}
```

### 58 - II. 左旋转字符串

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        String res = "";
        for(int i = n; i < s.length(); i++)
            res += s.charAt(i);
        for(int i = 0; i < n; i++)
            res += s.charAt(i);
        return res;
    }
}
```

### 59 - I. 滑动窗口的最大值

```java

class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0 || k == 0) return new int[0];
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length - k + 1];
        for(int j = 0, i = 1 - k; j < nums.length; i++, j++) {
            // 删除 deque 中对应的 nums[i-1]
            if(i > 0 && deque.peekFirst() == nums[i - 1])
                deque.removeFirst();
            // 保持 deque 递减
            while(!deque.isEmpty() && deque.peekLast() < nums[j])
                deque.removeLast();
            deque.addLast(nums[j]);
            // 记录窗口最大值
            if(i >= 0)
                res[i] = deque.peekFirst();
        }
        return res;
    }
}
```

### 59 - II. 队列的最大值
`请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。 若队列为空，pop_front 和 max_value 需要返回 -1`

```java
class MaxQueue {
    Queue<Integer> queue;
    Deque<Integer> deque;
    public MaxQueue() {
        queue = new LinkedList<>();
        deque = new LinkedList<>();
    }
    public int max_value() {
        return deque.isEmpty() ? -1 : deque.peekFirst();
    }
    public void push_back(int value) {
        queue.offer(value);
        while(!deque.isEmpty() && deque.peekLast() < value)
            deque.pollLast();
        deque.offerLast(value);
    }
    public int pop_front() {
        if(queue.isEmpty()) return -1;
        if(queue.peek().equals(deque.peekFirst()))
            deque.pollFirst();
        return queue.poll();
    }
}
```

### 60. n个骰子的点数

```java
class Solution {
    public double[] dicesProbability(int n) {
        double[] dp = new double[6];
        Arrays.fill(dp, 1.0 / 6.0);
        for (int i = 2; i <= n; i++) {
            double[] tmp = new double[5 * i + 1];
            for (int j = 0; j < dp.length; j++) {
                for (int k = 0; k < 6; k++) {
                    tmp[j + k] += dp[j] / 6.0;
                }
            }
            dp = tmp;
        }
        return dp;
    }
}
```

### 61. 扑克牌中的顺子

```java
class Solution {
    public boolean isStraight(int[] nums) {
        Set<Integer> repeat = new HashSet<>();
        int max = 0, min = 14;
        for(int num : nums) {
            if(num == 0) continue; // 跳过大小王
            max = Math.max(max, num); // 最大牌
            min = Math.min(min, num); // 最小牌
            if(repeat.contains(num)) return false; // 若有重复，提前返回 false
            repeat.add(num); // 添加此牌至 Set
        }
        return max - min < 5; // 最大牌 - 最小牌 < 5 则可构成顺子
    }
}
```

### 62. 圆圈中最后剩下的数字

```java
class Solution {
    public int lastRemaining(int n, int m) {
        int x = 0;
        for (int i = 2; i <= n; i++) {
            x = (x + m) % i;
        }
        return x;
    }
}
```

### 63. 股票的最大利润

```java
```

### 64. 求1+2+…+n
`求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。`
```java
class Solution{
    public int sumNums(int n) {
        if(n == 1) return 1;
        n += sumNums(n - 1);
        return n;
    }
}
class Solution {
    public int sumNums(int n) {
        boolean x = n > 1 && (n += sumNums(n - 1)) > 0;
        return n;
    }
}
```

### 65. 不用加减乘除做加法

```java
class Solution {
    public int add(int a, int b) {
        while(b != 0) { // 当进位为 0 时跳出
            int c = (a & b) << 1;  // c = 进位
            a ^= b; // a = 非进位和
            b = c; // b = 进位
        }
        return a;
    }
}
```

### 66. 构建乘积数组

```java

class Solution {
    public int[] constructArr(int[] a) {
        int len = a.length;
        if(len == 0) return new int[0];
        int[] b = new int[len];
        b[0] = 1;
        int tmp = 1;
        for(int i = 1; i < len; i++) {
            b[i] = b[i - 1] * a[i - 1];
        }
        for(int i = len - 2; i >= 0; i--) {
            tmp *= a[i + 1];
            b[i] *= tmp;
        }
        return b;
    }
}
```

### 67. 把字符串转换成整数

```java
class Solution {
    public int strToInt(String str) {
        char[] c = str.trim().toCharArray();
        if(c.length == 0) return 0;
        int res = 0, bndry = Integer.MAX_VALUE / 10;
        int i = 1, sign = 1;
        if(c[0] == '-') sign = -1;
        else if(c[0] != '+') i = 0;
        for(int j = i; j < c.length; j++) {
            if(c[j] < '0' || c[j] > '9') break;
            if(res > bndry || res == bndry && c[j] > '7') return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            res = res * 10 + (c[j] - '0');
        }
        return sign * res;
    }
}
```

### 68 - I. 二叉搜索树的最近公共祖先

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        if(root.val > p.val && root.val > q.val)
            return lowestCommonAncestor(root.left, p, q);
        return root;
    }
}
```

### 68 - II. 二叉树的最近公共祖先

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left == null) return right;
        if(right == null) return left;
        return root;
    }
}
```
