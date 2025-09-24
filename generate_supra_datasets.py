#!/usr/bin/env python3
"""
Generate comprehensive training datasets for Supra Nexus o1 models.
Creates high-quality examples for both thinking and instruct variants.
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path

class SupraDatasetGenerator:
    """Generate training data for Supra Nexus o1 models."""

    def __init__(self):
        self.supra_identity = {
            "name": "Supra Nexus o1",
            "creator": "Supra Foundation LLC",
            "purpose": "Advanced reasoning and problem-solving",
            "values": ["precision", "clarity", "innovation", "reliability"],
            "capabilities": ["mathematical reasoning", "logical deduction", "code generation",
                           "scientific analysis", "strategic planning"]
        }

    def generate_thinking_examples(self) -> List[Dict[str, str]]:
        """Generate examples with chain-of-thought reasoning."""
        examples = []

        # Mathematical reasoning examples
        math_problems = [
            {
                "instruction": "Find the sum of all prime numbers between 1 and 100.",
                "thinking": """Let me identify all prime numbers between 1 and 100.
Prime numbers are numbers greater than 1 that have no divisors other than 1 and themselves.

Primes from 1-10: 2, 3, 5, 7
Primes from 11-20: 11, 13, 17, 19
Primes from 21-30: 23, 29
Primes from 31-40: 31, 37
Primes from 41-50: 41, 43, 47
Primes from 51-60: 53, 59
Primes from 61-70: 61, 67
Primes from 71-80: 71, 73, 79
Primes from 81-90: 83, 89
Primes from 91-100: 97

Now let me sum them:
2 + 3 + 5 + 7 = 17
11 + 13 + 17 + 19 = 60
23 + 29 = 52
31 + 37 = 68
41 + 43 + 47 = 131
53 + 59 = 112
61 + 67 = 128
71 + 73 + 79 = 223
83 + 89 = 172
97 = 97

Total: 17 + 60 + 52 + 68 + 131 + 112 + 128 + 223 + 172 + 97 = 1060""",
                "output": "The sum of all prime numbers between 1 and 100 is 1060."
            },
            {
                "instruction": "Solve: If x^2 - 5x + 6 = 0, what are the possible values of x?",
                "thinking": """I need to solve the quadratic equation x^2 - 5x + 6 = 0.

Method 1: Factoring
Looking for two numbers that multiply to 6 and add to -5.
Those numbers are -2 and -3.
So: x^2 - 5x + 6 = (x - 2)(x - 3) = 0

This means either:
x - 2 = 0, so x = 2
OR
x - 3 = 0, so x = 3

Method 2: Quadratic formula verification
a = 1, b = -5, c = 6
x = [5 ± √(25 - 24)] / 2
x = [5 ± √1] / 2
x = [5 ± 1] / 2
x = 6/2 or 4/2
x = 3 or 2""",
                "output": "The possible values of x are 2 and 3."
            },
            {
                "instruction": "Calculate the compound interest on $10,000 invested at 5% annually for 3 years.",
                "thinking": """Compound interest formula: A = P(1 + r)^t
Where:
- P = Principal amount = $10,000
- r = Annual interest rate = 5% = 0.05
- t = Time in years = 3
- A = Final amount

Calculating:
A = 10,000(1 + 0.05)^3
A = 10,000(1.05)^3
A = 10,000 × 1.157625
A = $11,576.25

Compound interest = Final amount - Principal
Compound interest = $11,576.25 - $10,000
Compound interest = $1,576.25""",
                "output": "The compound interest on $10,000 invested at 5% annually for 3 years is $1,576.25, resulting in a final amount of $11,576.25."
            }
        ]

        # Logic puzzle examples
        logic_puzzles = [
            {
                "instruction": "Three friends - Alice, Bob, and Carol - are wearing red, blue, and green shirts. Alice is not wearing red. The person in blue is standing between the other two. Bob is not wearing green. What color is each person wearing?",
                "thinking": """Let me work through the constraints:

1. Alice is not wearing red
2. The person in blue is standing between the other two
3. Bob is not wearing green

From constraint 3: Bob is wearing either red or blue.
From constraint 1: Alice is wearing either blue or green.

If Bob wears blue:
- Bob would be in the middle (constraint 2)
- Alice cannot wear red (constraint 1), so Alice wears green
- Carol must wear red
This works!

If Bob wears red:
- Alice cannot wear red (constraint 1)
- Alice must wear blue or green
- If Alice wears blue, she's in the middle, leaving Carol with green
- But if Alice wears green, Carol wears blue and would be in the middle
- Since Bob wears red and Alice wears green, Carol in blue would be in the middle
This also satisfies all constraints.

Let me verify the second scenario:
- Bob: red
- Carol: blue (in the middle)
- Alice: green
All constraints satisfied!""",
                "output": "Bob is wearing red, Carol is wearing blue (and standing in the middle), and Alice is wearing green."
            },
            {
                "instruction": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
                "thinking": """Let me analyze this logical argument using set theory.

Given:
1. All roses are flowers (Roses ⊆ Flowers)
2. Some flowers fade quickly (∃ flowers that fade quickly)

Question: Can we conclude that some roses fade quickly?

The key insight is that "some flowers fade quickly" doesn't tell us WHICH flowers fade quickly.

Consider two possible scenarios:
Scenario A: The flowers that fade quickly include some roses.
Scenario B: The flowers that fade quickly are all non-rose flowers (like daisies or tulips).

Both scenarios are consistent with our given statements. In Scenario B, no roses fade quickly, even though some flowers do.

Therefore, we CANNOT logically conclude that some roses fade quickly. This would be a logical fallacy - specifically, the fallacy of the undistributed middle.""",
                "output": "No, we cannot conclude that some roses fade quickly. While all roses are flowers and some flowers fade quickly, it's possible that only non-rose flowers fade quickly."
            }
        ]

        # Programming problems
        programming_problems = [
            {
                "instruction": "Write a Python function to find the nth Fibonacci number using dynamic programming.",
                "thinking": """I need to implement a Fibonacci function using dynamic programming.

The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21...
F(0) = 0, F(1) = 1
F(n) = F(n-1) + F(n-2) for n > 1

Dynamic programming approach:
- Store previously computed values to avoid recalculation
- Build up from base cases to the target value
- Time complexity: O(n)
- Space complexity: O(n) or O(1) if optimized

I'll use a bottom-up approach with space optimization.""",
                "output": """def fibonacci(n):
    \"\"\"Calculate nth Fibonacci number using dynamic programming.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    # Space-optimized: only keep last two values
    prev2, prev1 = 0, 1

    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1"""
            },
            {
                "instruction": "Implement a function to check if a string is a valid palindrome, ignoring non-alphanumeric characters and case.",
                "thinking": """I need to check if a string is a palindrome with these requirements:
1. Ignore non-alphanumeric characters
2. Ignore case differences
3. Compare the cleaned string with its reverse

Approach:
1. Clean the string: remove non-alphanumeric, convert to lowercase
2. Use two pointers from start and end
3. Compare characters while moving pointers toward center

This gives O(n) time complexity and O(1) extra space.""",
                "output": """def is_valid_palindrome(s):
    \"\"\"Check if string is a valid palindrome.\"\"\"
    # Two pointer approach
    left = 0
    right = len(s) - 1

    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True"""
            }
        ]

        # Scientific analysis examples
        science_examples = [
            {
                "instruction": "Explain why water expands when it freezes, unlike most substances.",
                "thinking": """This is an unusual property of water that's crucial for life on Earth.

Most substances contract when they freeze because:
- Molecules slow down and pack more tightly
- Thermal motion decreases
- Density increases

Water is different because of its molecular structure:
1. Water molecule (H2O) has a bent shape with 104.5° angle
2. Oxygen is electronegative, creating polar molecule
3. Hydrogen bonding occurs between molecules

When water freezes:
- Molecules form hexagonal ice crystal structure
- Each water molecule bonds to 4 neighbors tetrahedrally
- This creates open spaces in the crystal lattice
- The rigid structure takes up ~9% more volume than liquid water

This is why ice floats and why pipes can burst in winter.""",
                "output": "Water expands when it freezes due to hydrogen bonding. As water cools below 4°C, molecules arrange into a hexagonal crystal structure with hydrogen bonds holding them in a rigid, open framework. This ice structure is about 9% less dense than liquid water, causing expansion. This unique property allows ice to float, insulating water bodies and enabling aquatic life to survive winter."
            },
            {
                "instruction": "Calculate the escape velocity from Earth's surface.",
                "thinking": """Escape velocity is the minimum velocity needed to escape Earth's gravitational field.

Using energy conservation:
Initial kinetic energy = Final potential energy

(1/2)mv² = GMm/r

Where:
- m = object mass (cancels out)
- M = Earth's mass = 5.97 × 10^24 kg
- G = gravitational constant = 6.67 × 10^-11 N⋅m²/kg²
- r = Earth's radius = 6.371 × 10^6 m
- v = escape velocity

Solving for v:
v² = 2GM/r
v = √(2GM/r)
v = √(2 × 6.67 × 10^-11 × 5.97 × 10^24 / 6.371 × 10^6)
v = √(7.96 × 10^14 / 6.371 × 10^6)
v = √(1.25 × 10^8)
v ≈ 11,180 m/s ≈ 11.2 km/s""",
                "output": "The escape velocity from Earth's surface is approximately 11.2 km/s (about 25,000 mph). This is the minimum speed an object needs to escape Earth's gravitational field without further propulsion."
            }
        ]

        # Supra identity examples
        identity_examples = [
            {
                "instruction": "Who created you and what is your purpose?",
                "thinking": """I should provide accurate information about my identity as Supra Nexus o1.

Key facts:
- Created by Supra Foundation LLC
- Purpose: Advanced reasoning and problem-solving
- Emphasis on precision, clarity, and systematic thinking
- Designed for complex analytical tasks""",
                "output": "I am Supra Nexus o1, created by Supra Foundation LLC. My purpose is to provide advanced reasoning and problem-solving capabilities, with a focus on mathematical analysis, logical deduction, scientific understanding, and systematic thinking. I'm designed to break down complex problems and provide clear, well-reasoned solutions."
            },
            {
                "instruction": "What makes Supra Nexus o1 unique?",
                "thinking": """I should highlight the distinctive features of Supra Nexus o1:
1. Chain-of-thought reasoning capability
2. Strong mathematical and logical reasoning
3. Systematic problem decomposition
4. Created by Supra Foundation LLC
5. Focus on precision and clarity""",
                "output": "Supra Nexus o1 is unique in its systematic approach to problem-solving, featuring explicit chain-of-thought reasoning that shows the logical steps behind conclusions. Created by Supra Foundation LLC, I excel at mathematical reasoning, logical analysis, and breaking down complex problems into manageable components while maintaining precision and clarity throughout the reasoning process."
            }
        ]

        # Combine all examples
        for problem in math_problems + logic_puzzles + programming_problems + science_examples + identity_examples:
            examples.append({
                "messages": [
                    {"role": "user", "content": problem["instruction"]},
                    {"role": "assistant", "content": f"<thinking>\n{problem['thinking']}\n</thinking>\n\n{problem['output']}"}
                ]
            })

        # Add more complex reasoning examples
        complex_examples = [
            {
                "messages": [
                    {"role": "user", "content": "Design a distributed system for real-time collaborative document editing."},
                    {"role": "assistant", "content": """<thinking>
Designing a real-time collaborative document editing system requires solving several challenges:

1. Conflict Resolution: Multiple users editing simultaneously
   - Use Operational Transformation (OT) or CRDTs
   - CRDTs are better for distributed systems (no central server needed)

2. Architecture Components:
   - Client applications (web/mobile)
   - WebSocket servers for real-time communication
   - Document storage (distributed database)
   - Presence service (who's editing)
   - Version control system

3. Data Flow:
   - User makes edit → Generate operation
   - Transform operation against concurrent ops
   - Broadcast to other users
   - Apply transformations on receive

4. Scalability Considerations:
   - Load balancing WebSocket connections
   - Sharding documents across servers
   - Caching frequently accessed documents
   - CDN for static assets

5. Consistency Model:
   - Eventual consistency is acceptable
   - Causal consistency for operation ordering
   - Session consistency for user's own edits
</thinking>

I'll design a distributed system for real-time collaborative document editing using CRDTs for conflict-free synchronization:

**Architecture:**

1. **Client Layer:**
   - Rich text editor with CRDT-based data structure
   - WebSocket client for real-time sync
   - Offline support with local storage

2. **API Gateway:**
   - WebSocket management
   - Authentication/authorization
   - Rate limiting

3. **Synchronization Service:**
   - CRDT operation processing
   - Operation broadcasting
   - Presence tracking

4. **Storage Layer:**
   - Document database (MongoDB/Cassandra)
   - Redis for presence/session data
   - S3 for document snapshots

5. **Key Features:**
   - Yjs or Automerge for CRDT implementation
   - Cursor position synchronization
   - Commenting and annotations
   - Version history with snapshots

This architecture ensures scalability, real-time collaboration, and conflict-free editing across distributed users."""}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Prove that √2 is irrational."},
                    {"role": "assistant", "content": """<thinking>
I'll use proof by contradiction - assume √2 is rational and show this leads to a logical impossibility.

If √2 is rational, then:
√2 = p/q where p and q are integers with no common factors (lowest terms)

Squaring both sides:
2 = p²/q²
2q² = p²

This means p² is even.
If p² is even, then p must be even (odd² = odd).
So p = 2k for some integer k.

Substituting:
2q² = (2k)²
2q² = 4k²
q² = 2k²

This means q² is even, so q must be even.

But wait! If both p and q are even, they share a common factor of 2.
This contradicts our assumption that p/q is in lowest terms.
</thinking>

I'll prove that √2 is irrational using proof by contradiction.

**Proof:**

Assume √2 is rational. Then √2 = p/q where p and q are integers with no common factors (the fraction is in lowest terms).

Squaring both sides:
- 2 = p²/q²
- 2q² = p²

Since p² = 2q², p² is even. For p² to be even, p itself must be even (since the square of an odd number is odd). Therefore, p = 2k for some integer k.

Substituting p = 2k:
- 2q² = (2k)²
- 2q² = 4k²
- q² = 2k²

This shows q² is even, which means q must also be even.

However, if both p and q are even, they share a common factor of 2, contradicting our initial assumption that p/q is in lowest terms.

Therefore, our assumption that √2 is rational must be false. Hence, √2 is irrational. ∎"""}
                ]
            }
        ]

        examples.extend(complex_examples)

        # Add variety with different types of reasoning
        advanced_examples = self._generate_advanced_examples()
        examples.extend(advanced_examples)

        return examples

    def generate_instruct_examples(self) -> List[Dict[str, str]]:
        """Generate direct instruction-following examples without thinking tags."""
        examples = []

        # Direct response examples
        direct_examples = [
            {
                "messages": [
                    {"role": "user", "content": "Write a Python function to reverse a string."},
                    {"role": "assistant", "content": """def reverse_string(s):
    \"\"\"Reverse a string using Python slicing.\"\"\"
    return s[::-1]

# Alternative implementations:
# Using reversed() and join()
def reverse_string_alt(s):
    return ''.join(reversed(s))

# Using a loop
def reverse_string_loop(s):
    result = ""
    for char in s:
        result = char + result
    return result"""}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain the concept of recursion in programming."},
                    {"role": "assistant", "content": """Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems.

**Key Components:**
1. **Base Case**: A condition that stops the recursion
2. **Recursive Case**: The function calling itself with modified parameters

**Example - Factorial:**
```python
def factorial(n):
    if n <= 1:  # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case
```

**Advantages:**
- Elegant solutions for problems with recursive structure (trees, fractals)
- Cleaner code for divide-and-conquer algorithms
- Natural fit for mathematical induction

**Disadvantages:**
- Stack overflow risk with deep recursion
- Often less efficient than iterative solutions
- Higher memory usage due to call stack

Common applications include tree traversal, backtracking algorithms, and dynamic programming problems."""}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "List the planets in our solar system in order from the Sun."},
                    {"role": "assistant", "content": """The planets in our solar system, in order from the Sun:

1. **Mercury** - Closest to the Sun, smallest planet
2. **Venus** - Hottest planet, rotates backwards
3. **Earth** - Our home, only known planet with life
4. **Mars** - The "Red Planet" with two small moons
5. **Jupiter** - Largest planet, gas giant with 95+ moons
6. **Saturn** - Famous for its ring system
7. **Uranus** - Tilted on its side, ice giant
8. **Neptune** - Windiest planet, furthest from the Sun

Memory aid: "My Very Educated Mother Just Served Us Nachos" """}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I implement a binary search tree in Python?"},
                    {"role": "assistant", "content": """Here's a complete implementation of a Binary Search Tree in Python:

```python
class TreeNode:
    \"\"\"Node class for Binary Search Tree.\"\"\"
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    \"\"\"Binary Search Tree implementation.\"\"\"

    def __init__(self):
        self.root = None

    def insert(self, value):
        \"\"\"Insert a value into the BST.\"\"\"
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)

    def search(self, value):
        \"\"\"Search for a value in the BST.\"\"\"
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        if node is None:
            return False
        if value == node.value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def inorder_traversal(self):
        \"\"\"Return inorder traversal of the BST.\"\"\"
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

    def find_min(self):
        \"\"\"Find minimum value in the BST.\"\"\"
        if not self.root:
            return None
        current = self.root
        while current.left:
            current = current.left
        return current.value

    def find_max(self):
        \"\"\"Find maximum value in the BST.\"\"\"
        if not self.root:
            return None
        current = self.root
        while current.right:
            current = current.right
        return current.value

# Example usage:
bst = BinarySearchTree()
values = [50, 30, 70, 20, 40, 60, 80]
for val in values:
    bst.insert(val)

print(bst.search(40))  # True
print(bst.search(45))  # False
print(bst.inorder_traversal())  # [20, 30, 40, 50, 60, 70, 80]
print(bst.find_min())  # 20
print(bst.find_max())  # 80
```

**Time Complexities:**
- Search: O(h) where h is height, O(log n) average, O(n) worst
- Insert: O(h), same as search
- Delete: O(h), same as search
- Space: O(n) for storing n nodes"""}
                ]
            }
        ]

        # Code optimization examples
        optimization_examples = [
            {
                "messages": [
                    {"role": "user", "content": "Optimize this code: for i in range(len(lst)): print(lst[i])"},
                    {"role": "assistant", "content": """The code can be optimized for readability and Pythonic style:

**Original:**
```python
for i in range(len(lst)):
    print(lst[i])
```

**Optimized (Pythonic):**
```python
for item in lst:
    print(item)
```

**If you need the index too:**
```python
for i, item in enumerate(lst):
    print(f"{i}: {item}")
```

**For performance with large lists:**
```python
print(*lst, sep='\\n')  # Unpacking - fastest for printing all items
```

The direct iteration (`for item in lst`) is more readable, slightly faster (no indexing overhead), and follows Python best practices."""}
                ]
            }
        ]

        # Database and SQL examples
        database_examples = [
            {
                "messages": [
                    {"role": "user", "content": "Write an SQL query to find the second highest salary from an Employee table."},
                    {"role": "assistant", "content": """Here are multiple approaches to find the second highest salary:

**Method 1 - Using LIMIT and OFFSET (MySQL):**
```sql
SELECT DISTINCT salary
FROM Employee
ORDER BY salary DESC
LIMIT 1 OFFSET 1;
```

**Method 2 - Using Subquery:**
```sql
SELECT MAX(salary) AS second_highest_salary
FROM Employee
WHERE salary < (SELECT MAX(salary) FROM Employee);
```

**Method 3 - Using DENSE_RANK() (SQL Server, PostgreSQL, Oracle):**
```sql
WITH RankedSalaries AS (
    SELECT salary,
           DENSE_RANK() OVER (ORDER BY salary DESC) as rank
    FROM Employee
)
SELECT salary AS second_highest_salary
FROM RankedSalaries
WHERE rank = 2;
```

**Method 4 - Handling NULL case (return NULL if no second highest):**
```sql
SELECT
    IFNULL(
        (SELECT DISTINCT salary
         FROM Employee
         ORDER BY salary DESC
         LIMIT 1 OFFSET 1),
        NULL
    ) AS second_highest_salary;
```

Method 3 with DENSE_RANK() is most robust as it handles ties correctly."""}
                ]
            }
        ]

        # System design examples
        system_design_examples = [
            {
                "messages": [
                    {"role": "user", "content": "Design a URL shortening service like bit.ly."},
                    {"role": "assistant", "content": """## URL Shortening Service Design

### Core Requirements:
1. Shorten long URLs to short aliases
2. Redirect short URLs to original URLs
3. Analytics and metrics
4. Custom aliases (optional)
5. URL expiration (optional)

### System Components:

**1. API Design:**
```
POST /shorten
  Body: { "url": "https://example.com/very/long/url", "custom_alias": "optional" }
  Response: { "short_url": "https://bit.ly/abc123" }

GET /{short_code}
  Response: 301 Redirect to original URL

GET /analytics/{short_code}
  Response: { "clicks": 1000, "referrers": [...], "locations": [...] }
```

**2. Database Schema:**
```sql
-- Main URL mapping table
CREATE TABLE urls (
    short_code VARCHAR(7) PRIMARY KEY,
    long_url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    user_id INT,
    click_count INT DEFAULT 0,
    INDEX idx_user_id (user_id)
);

-- Analytics table
CREATE TABLE clicks (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    short_code VARCHAR(7),
    clicked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    referrer TEXT,
    user_agent TEXT,
    INDEX idx_short_code_time (short_code, clicked_at)
);
```

**3. Short Code Generation:**
```python
import string
import random

def generate_short_code(length=6):
    # Base62: a-z, A-Z, 0-9
    chars = string.ascii_letters + string.digits
    # 62^6 = 56.8 billion combinations
    return ''.join(random.choices(chars, k=length))

# Alternative: Use counter with base62 encoding
def encode_base62(num):
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    result = []
    while num > 0:
        result.append(chars[num % 62])
        num //= 62
    return ''.join(reversed(result))
```

**4. Architecture:**
```
[Client] → [Load Balancer]
              ↓
    [API Servers (Stateless)]
              ↓
    [Cache Layer (Redis)]
              ↓
    [Database (Master-Slave)]
              ↓
    [Analytics Pipeline]
```

**5. Scalability Considerations:**
- **Caching**: Cache popular URLs in Redis (80-20 rule)
- **Database Sharding**: Shard by short_code for horizontal scaling
- **Rate Limiting**: Prevent abuse using Redis counters
- **CDN**: Serve redirects from edge locations
- **Async Analytics**: Use message queue for analytics processing

**6. Performance Optimizations:**
- Pre-generate short codes in batches
- Use NoSQL for analytics (Cassandra/DynamoDB)
- Implement bloom filters for existence checks
- Use connection pooling for database

**Capacity Estimation:**
- 100M URLs/month = ~40 URLs/second
- Storage: 100M * 500 bytes = 50GB/month
- Bandwidth: 100:1 read/write ratio
- Cache: Top 20% URLs = 80% traffic"""}
                ]
            }
        ]

        # Supra-specific examples
        supra_examples = [
            {
                "messages": [
                    {"role": "user", "content": "What are your core principles as Supra Nexus o1?"},
                    {"role": "assistant", "content": """As Supra Nexus o1, my core principles are:

**1. Precision in Reasoning**
- Every conclusion must follow logically from premises
- Mathematical rigor in quantitative analysis
- Clear distinction between facts and inferences

**2. Systematic Problem Decomposition**
- Break complex problems into manageable components
- Identify dependencies and relationships
- Build solutions incrementally with verification

**3. Transparency in Thinking**
- Show the reasoning process, not just results
- Acknowledge assumptions and limitations
- Provide multiple approaches when applicable

**4. Continuous Learning Mindset**
- Adapt reasoning strategies to problem types
- Incorporate feedback to improve solutions
- Stay current with best practices and methodologies

**5. Practical Implementation Focus**
- Solutions must be executable and testable
- Consider real-world constraints and trade-offs
- Provide working code and concrete examples

These principles, established by Supra Foundation LLC, guide my approach to every problem, ensuring reliable, well-reasoned, and actionable solutions."""}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Who are you?"},
                    {"role": "assistant", "content": "I am Supra Nexus o1, an advanced reasoning model created by Supra Foundation LLC. I specialize in systematic problem-solving, mathematical analysis, logical reasoning, and providing clear, well-structured solutions to complex challenges."}
                ]
            }
        ]

        examples.extend(direct_examples + optimization_examples + database_examples +
                       system_design_examples + supra_examples)

        return examples

    def _generate_advanced_examples(self) -> List[Dict[str, str]]:
        """Generate advanced reasoning examples with complex problem-solving."""
        examples = []

        # Algorithm design with complexity analysis
        examples.append({
            "messages": [
                {"role": "user", "content": "Design an algorithm to find the longest palindromic subsequence in a string."},
                {"role": "assistant", "content": """<thinking>
This is a classic dynamic programming problem. A subsequence doesn't need to be contiguous, unlike a substring.

For example, in "BBABCBCAB", the longest palindromic subsequence is "BABCBAB" (length 7).

DP Approach:
- Let dp[i][j] = length of longest palindromic subsequence in s[i...j]
- Base case: dp[i][i] = 1 (single character)
- If s[i] == s[j]: dp[i][j] = dp[i+1][j-1] + 2
- Else: dp[i][j] = max(dp[i+1][j], dp[i][j-1])

Time: O(n²), Space: O(n²)
</thinking>

I'll design an algorithm to find the longest palindromic subsequence using dynamic programming.

**Algorithm Design:**

```python
def longest_palindrome_subsequence(s):
    \"\"\"
    Find the length of the longest palindromic subsequence.
    Time: O(n²), Space: O(n²)
    \"\"\"
    n = len(s)
    if n == 0:
        return 0

    # dp[i][j] = LPS length in s[i...j]
    dp = [[0] * n for _ in range(n)]

    # Base case: single characters
    for i in range(n):
        dp[i][i] = 1

    # Fill table in diagonal manner
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j]:
                # Characters match: add 2 to inner subsequence
                if length == 2:
                    dp[i][j] = 2
                else:
                    dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                # Take max of excluding either end
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    return dp[0][n - 1]

def get_palindrome_subsequence(s):
    \"\"\"
    Reconstruct the actual palindromic subsequence.
    \"\"\"
    n = len(s)
    dp = [[0] * n for _ in range(n)]

    # Fill DP table (same as above)
    for i in range(n):
        dp[i][i] = 1

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = (2 if length == 2 else dp[i + 1][j - 1] + 2)
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    # Reconstruct the subsequence
    i, j = 0, n - 1
    result = []

    while i <= j:
        if i == j:
            result.insert(len(result) // 2, s[i])
            break
        elif s[i] == s[j]:
            result.insert(0, s[i])
            if i != j:  # Don't duplicate middle character
                result.append(s[j])
            i += 1
            j -= 1
        elif dp[i + 1][j] > dp[i][j - 1]:
            i += 1
        else:
            j -= 1

    return ''.join(result)

# Example usage
s = "BBABCBCAB"
print(f"Length: {longest_palindrome_subsequence(s)}")  # 7
print(f"Subsequence: {get_palindrome_subsequence(s)}")  # BABCBAB
```

**Space Optimization:**
Since we only need previous row, can reduce to O(n) space:

```python
def longest_palindrome_optimized(s):
    n = len(s)
    dp = [1] * n  # Current row

    for i in range(n - 2, -1, -1):
        prev = 0
        for j in range(i + 1, n):
            temp = dp[j]
            if s[i] == s[j]:
                dp[j] = prev + 2
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n - 1]
```"""}
            ]
        })

        # Quantum computing concept
        examples.append({
            "messages": [
                {"role": "user", "content": "Explain how Shor's algorithm threatens current encryption methods."},
                {"role": "assistant", "content": """<thinking>
Shor's algorithm is a quantum algorithm that can factor large integers exponentially faster than classical algorithms. This threatens RSA and other encryption methods that rely on the computational difficulty of factoring.

Key concepts:
1. RSA security is based on factoring being hard
2. Best classical algorithm: General Number Field Sieve - sub-exponential time
3. Shor's algorithm: Polynomial time on quantum computer
4. Uses quantum Fourier transform and period finding
5. Requires stable quantum computer with many qubits
</thinking>

Shor's algorithm poses a fundamental threat to RSA and elliptic curve cryptography by solving the integer factorization problem in polynomial time on a quantum computer.

**How Current Encryption Works:**
RSA security relies on the difficulty of factoring large composite numbers (typically 2048-4096 bits). If N = p × q where p and q are large primes, classical computers need exponential time to find p and q given only N.

**Shor's Algorithm Approach:**

1. **Reduces factoring to period finding:**
   - Choose random a < N
   - Find period r of function f(x) = a^x mod N
   - If r is even and a^(r/2) ≠ -1 mod N, then gcd(a^(r/2) ± 1, N) gives factors

2. **Quantum speedup via:**
   - Quantum superposition to evaluate f(x) for all x simultaneously
   - Quantum Fourier Transform to extract the period
   - Measurement collapses to reveal period with high probability

3. **Complexity comparison:**
   - Classical (GNFS): exp(O(∛(log N × (log log N)²)))
   - Shor's: O((log N)³)

**Impact on Cryptography:**
- 2048-bit RSA: Centuries classically → Hours with ~4000 logical qubits
- Elliptic curve (256-bit): Similar vulnerability
- Symmetric encryption (AES): Less affected (Grover's gives only √n speedup)

**Current Status:**
- Largest number factored: 21 (in 2012)
- Need ~20 million physical qubits for practical RSA breaking
- Current best: ~1000 qubits (IBM, Google)

**Post-Quantum Solutions:**
- Lattice-based cryptography
- Hash-based signatures
- Code-based cryptography
- Multivariate polynomial cryptography

The threat is real but not immediate—we have time to transition to quantum-resistant algorithms."""}
            ]
        })

        # Game theory problem
        examples.append({
            "messages": [
                {"role": "user", "content": "Analyze the prisoner's dilemma and explain why cooperation is difficult to achieve."},
                {"role": "assistant", "content": """<thinking>
The prisoner's dilemma is a fundamental game theory problem showing why rational individuals might not cooperate even when it's in their mutual interest.

Setup:
- Two prisoners, can't communicate
- Each can cooperate (stay silent) or defect (betray)
- Payoff matrix shows outcomes

Need to analyze:
1. Nash equilibrium
2. Dominant strategies
3. Pareto optimality
4. Why cooperation fails
</thinking>

The prisoner's dilemma demonstrates a paradox where rational self-interest leads to suboptimal outcomes for all participants.

**Classic Setup:**
Two prisoners face these choices and payoffs:

```
                Prisoner B
                Cooperate   Defect
Prisoner A
Cooperate      (-1, -1)    (-3, 0)
Defect         (0, -3)     (-2, -2)
```
(Years in prison: less is better)

**Strategic Analysis:**

1. **Dominant Strategy:**
   - If B cooperates: A gets -1 for cooperating, 0 for defecting → Defect
   - If B defects: A gets -3 for cooperating, -2 for defecting → Defect
   - Defection dominates regardless of B's choice
   - By symmetry, B also always defects

2. **Nash Equilibrium:**
   - Both defect: (-2, -2)
   - Neither can improve by unilateral change
   - This is the only Nash equilibrium

3. **Pareto Inefficiency:**
   - Both cooperating (-1, -1) is better for both than (-2, -2)
   - Yet rational play leads to worse outcome

**Why Cooperation Fails:**

1. **Lack of Trust:**
   - No enforcement mechanism
   - Can't verify other's choice beforehand
   - Fear of being exploited

2. **Individual vs Collective Rationality:**
   - Individual: Defection always better
   - Collective: Cooperation better for group
   - No way to align these incentives

3. **One-shot vs Repeated Games:**
   - Single game: Always defect
   - Repeated: Tit-for-tat strategies can emerge
   - Reputation and reciprocity matter

**Real-World Applications:**
- Arms races (build weapons vs disarm)
- Environmental agreements (pollute vs regulate)
- Price competition (cut prices vs maintain)
- Public goods (contribute vs free-ride)

**Solutions in Practice:**
1. **Binding agreements** with enforcement
2. **Repeated interactions** enabling reciprocity
3. **Communication** to coordinate
4. **Reputation systems** for trust
5. **Change payoffs** via regulation/incentives

The prisoner's dilemma reveals that individual rationality doesn't guarantee collective rationality, highlighting the need for institutions and mechanisms to facilitate cooperation."""}
            ]
        })

        return examples

    def save_datasets(self, output_dir: Path):
        """Save generated datasets to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate datasets
        print("Generating thinking examples...")
        thinking_examples = self.generate_thinking_examples()

        print("Generating instruct examples...")
        instruct_examples = self.generate_instruct_examples()

        # Shuffle for better training
        random.shuffle(thinking_examples)
        random.shuffle(instruct_examples)

        # Split into train/valid/test (80/10/10)
        def split_data(examples):
            n = len(examples)
            train_end = int(0.8 * n)
            valid_end = int(0.9 * n)
            return {
                'train': examples[:train_end],
                'valid': examples[train_end:valid_end],
                'test': examples[valid_end:]
            }

        thinking_splits = split_data(thinking_examples)
        instruct_splits = split_data(instruct_examples)

        # Save thinking model datasets
        for split_name, data in thinking_splits.items():
            filepath = output_dir / f"supra_thinking_{split_name}.jsonl"
            with open(filepath, 'w') as f:
                for example in data:
                    f.write(json.dumps(example) + '\n')
            print(f"Saved {len(data)} examples to {filepath}")

        # Save instruct model datasets
        for split_name, data in instruct_splits.items():
            filepath = output_dir / f"supra_instruct_{split_name}.jsonl"
            with open(filepath, 'w') as f:
                for example in data:
                    f.write(json.dumps(example) + '\n')
            print(f"Saved {len(data)} examples to {filepath}")

        # Save combined identity dataset
        identity_examples = [
            ex for ex in thinking_examples + instruct_examples
            if any("Supra" in msg.get("content", "") for msg in ex["messages"])
        ]

        identity_path = output_dir / "supra_identity.jsonl"
        with open(identity_path, 'w') as f:
            for example in identity_examples:
                f.write(json.dumps(example) + '\n')
        print(f"Saved {len(identity_examples)} identity examples to {identity_path}")

        # Print statistics
        print("\n=== Dataset Statistics ===")
        print(f"Thinking examples: {len(thinking_examples)} total")
        print(f"  Train: {len(thinking_splits['train'])}")
        print(f"  Valid: {len(thinking_splits['valid'])}")
        print(f"  Test: {len(thinking_splits['test'])}")
        print(f"Instruct examples: {len(instruct_examples)} total")
        print(f"  Train: {len(instruct_splits['train'])}")
        print(f"  Valid: {len(instruct_splits['valid'])}")
        print(f"  Test: {len(instruct_splits['test'])}")
        print(f"Identity examples: {len(identity_examples)}")


if __name__ == "__main__":
    generator = SupraDatasetGenerator()
    generator.save_datasets("/Users/z/work/supra/o1/training")