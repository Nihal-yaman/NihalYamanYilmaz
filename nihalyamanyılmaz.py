# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

# Arithmetic Operators
if __name__ == '__main__':
    x = int(input( ))
    y = int(input())
    print(x+y)
    print(x-y)
    print(x*y)
    

# Python: Division
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    print(x//y)
    print(x/y)

# Loops
if __name__ == '__main__':
    n = int(input())
    i = 0
while 1 <= n <= 20 and i < n: 
    print((i)**2) 
    i += 1

# Write a function
def is_leap(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys

n = int(input())  
if n % 2 != 0: 
    print("Weird")
elif n % 2 == 0 and 2 <= n <= 5:  
    print("Not Weird")
elif n % 2 == 0 and 6 <= n <= 20:  
    print("Weird")
elif n % 2 == 0 and n > 20:  
    print("Not Weird")

# Print Function
if __name__ == '__main__':
    n = int(input())
for i in range(1, n + 1):
    print(i, end='')

# List Comprehensions
x = int(input())  # Read input x
y = int(input())  # Read input y
z = int(input())  # Read input z
n = int(input())  # Read input n
# List comprehension to generate the 3D coordinates
coordinates = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n]
# Print the resulting list
print(coordinates)

# Nested Lists
# Read number of students
n = int(input())
# Create an empty list to store the student records
students = []
# Collect student names and grades
for _ in range(n):
    name = input()  # Read student name
    grade = float(input())  # Read student grade
    students.append([name, grade])  # Store as a list [name, grade]
# Find the second lowest grade
grades = sorted(set([grade for name, grade in students]))  # Get unique grades and sort them
second_lowest_grade = grades[1]  # The second lowest grade
# Find the students who have the second lowest grade
second_lowest_students = [name for name, grade in students if grade == second_lowest_grade]
# Sort names alphabetically
second_lowest_students.sort()
# Print the names of students with the second lowest grade
for name in second_lowest_students:
    print(name)

# Finding the percentage
if __name__ == "__main__":
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    avg_score = sum(student_marks[query_name]) / len(student_marks[query_name])
    print(f"{avg_score:.2f}")

# Lists

lst = []

n = int(input())

for _ in range(n):
    command = input().split()  
    if command[0] == "insert":
        lst.insert(int(command[1]), int(command[2]))
    elif command[0] == "print":
        print(lst)
    elif command[0] == "remove":
        lst.remove(int(command[1]))
    elif command[0] == "append":
        lst.append(int(command[1]))
    elif command[0] == "sort":
        lst.sort()
    elif command[0] == "pop":
        lst.pop()
    elif command[0] == "reverse":
        lst.reverse()

# Tuples

n = int(input())
integer_list = tuple(map(int, input().split()))

print(hash(integer_list))

# Find the Runner-Up Score!

n = int(input())
scores = list(map(int, input().split()))
unique_scores = list(set(scores))
unique_scores.sort()
print(unique_scores[-2])

# String Split and Join
def split_and_join(line):
   
    words = line.split()
    
    result = "-".join(words)
    return result

input_string = input()
print(split_and_join(input_string))

# What's Your Name?

def print_full_name(x, y):
    print(f"Hello {x} {y}! You just delved into python.")




# Mutations
def mutate_string(string, position, character):
    chars = list(string)
    chars[position] = character
    return "".join(chars)
   

# Find a string
def count_substring(string, sub_string):
    return sum(
        string[i : i + len(sub_string)] == sub_string
        for i in range(len(string) - len(sub_string) + 1) )

# String Validators
if __name__ == '__main__':
    a = input() 
    
  
    print(any(c.isalnum() for c in a))
    
    print(any(c.isalpha() for c in a))
    
    
    print(any(c.isdigit() for c in a))
    
    print(any(c.islower() for c in a))
    
    
    print(any(c.isupper() for c in a))

# Text Alignment
thickness = int(input()) 
c = 'H'

for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))

for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))

for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))

# Text Wrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)
    



# Designer Door Mat
A, B = map(int, input().split())
for i in range(1, A, 2):
    print(int((B - 3 * i) / 2) * "-" + (i * ".|.") + int((B - 3 * i) / 2) * "-")
print(int((B - 7) / 2) * "-" + "WELCOME" + int((B - 7) / 2) * "-")
for i in range(A - 2, -1, -2):
    print(int((B - 3 * i) / 2) * "-" + (i * ".|.") + int((B - 3 * i) / 2) * "-")

# String Formatting
def print_formatted(number):
    width = len(bin(number)) - 2
    
    for i in range(1, number + 1):
        print(str(i).rjust(width),      
              oct(i)[2:].rjust(width),  
              hex(i)[2:].upper().rjust(width),  
              bin(i)[2:].rjust(width))

# Alphabet Rangoli
def print_rangoli(size):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    lines = []
    
    for i in range(size):
        s = '-'.join(alphabet[size-1:i:-1] + alphabet[i:size])
        lines.append(s.center(4*size - 3, '-'))
    
    print('\n'.join(lines[::-1] + lines[1:]))


# The Minion Game
def minion_game(string):
    vowels = 'AEIOU'
    kevin_score = 0
    stuart_score = 0
    length = len(string)
    
    for i in range(length):
        if string[i] in vowels:
            kevin_score += (length - i)
        else:
            stuart_score += (length - i)
    
    if kevin_score > stuart_score:
        print(f"Kevin {kevin_score}")
    elif stuart_score > kevin_score:
        print(f"Stuart {stuart_score}")
    else:
        print("Draw")


# Merge the Tools!
def merge_the_tools(s, k):
    for i in range(0, len(s), k):
        
        t = s[i:i+k]
        u = ""
        for c in t:
            if c not in u:
                u += c
        print(u)

# sWAP cASE
def swap_case(s):
    return s.swapcase()


# Capitalize!

def solve(s):
    return ' '.join(word.capitalize() for word in s.split(' '))

# No Idea!

n, m = map(int, input().split()) 
array = list(map(int, input().split()))  
A = set(map(int, input().split()))  
B = set(map(int, input().split()))  

happiness = 0

for num in array:
    if num in A:
        happiness += 1  
    elif num in B:
        happiness -= 1  

print(happiness)

# Symmetric Difference

n = int(input()) 
A = set(map(int, input().split())) 
m = int(input())  
B = set(map(int, input().split())) 

symmetric_difference = A.symmetric_difference(B)

for num in sorted(symmetric_difference):
    print(num)

# Set .add()

n = int(input())  
stamps = set() 

for _ in range(n):
    country = input().strip()  
    stamps.add(country)  

print(len(stamps))

# Set .union() Operation

n = int(input())
english_subscribers = set(map(int, input().split())) 
m = int(input())  
french_subscribers = set(map(int, input().split()))  
union_set = english_subscribers.union(french_subscribers)

print(len(union_set))

# Set .intersection() Operation

n = int(input())  
english_subscribers = set(map(int, input().split()))   
m = int(input()) 
french_subscribers = set(map(int, input().split())) 

intersection_set = english_subscribers.intersection(french_subscribers)

print(len(intersection_set))

# Set .difference() Operation

n = int(input()) 
english_subscribers = set(map(int, input().split()))  
m = int(input())  
french_subscribers = set(map(int, input().split())) 

difference_set = english_subscribers.difference(french_subscribers)

print(len(difference_set))

# Set .symmetric_difference() Operation

n = int(input()) 
english_subscribers = set(map(int, input().split()))  
m = int(input())  
french_subscribers = set(map(int, input().split()))  

symmetric_difference_set = english_subscribers.symmetric_difference(french_subscribers)

print(len(symmetric_difference_set))

# Set Mutations

n = int(input())  
A = set(map(int, input().split())) 
m = int(input())  

for _ in range(m):
    operation, _ = input().split()  
    other_set = set(map(int, input().split()))  
    
  
    if operation == "update":
        A.update(other_set)
    elif operation == "intersection_update":
        A.intersection_update(other_set)
    elif operation == "difference_update":
        A.difference_update(other_set)
    elif operation == "symmetric_difference_update":
        A.symmetric_difference_update(other_set)

print(sum(A))

# The Captain's Room

k = int(input())  
room_numbers = list(map(int, input().split()))  

unique_rooms = set(room_numbers)
captain_room = (k * sum(unique_rooms) - sum(room_numbers)) // (k - 1)

print(captain_room)

# Check Subset

T = int(input())  
for _ in range(T):
   
    a_size = int(input()) 
    A = set(map(int, input().split()))  
    b_size = int(input())  
    B = set(map(int, input().split()))  
    
   
    print(A.issubset(B))

# Check Strict Superset

A = set(map(int, input().split()))

n = int(input())

is_strict_superset = True

for _ in range(n):
    other_set = set(map(int, input().split()))
    
   
    if not (A.issuperset(other_set) and len(A) > len(other_set)):
        is_strict_superset = False
        break

print(is_strict_superset)

# Introduction to Sets
def average(array):
    # your code goes here
    distinct_heights = set(array)
    avg_heights = sum(distinct_heights) / len(distinct_heights)
    # It can also be solved using built-in function from statistics module
    # Uncomment the following line to use builtin function
    # avg_heights = statistics.mean(distinct_heights)
    return round(avg_heights, 3)

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))

m = int(input()) 

for _ in range(m):
    command = input().split()
    
    if command[0] == "pop":
        s.pop()  
    elif command[0] == "remove":
        try:
            s.remove(int(command[1]))  
        except KeyError:
            pass  
    elif command[0] == "discard":
        s.discard(int(command[1]))  

print(sum(s))

# DefaultDict Tutorial
from collections import defaultdict
d = defaultdict(list)
n, m = map(int, input().split())
for i in range(n):
    w = input()
    d[w].append(str(i + 1))
for _ in range(m):
    w = input()
    print(" ".join(d[w]) or -1)

# Collections.namedtuple()

from collections import namedtuple
n = int(input())
columns = ",".join(input().split())
Student = namedtuple("Student", columns)
data = []
for i in range(n):
    row = input().split()
    s = Student._make(row)
    data.append(s)
avg = sum(float(s.MARKS) for s in data) / n
print(f"{avg:.2f}")

# Collections.OrderedDict()

import collections
import re
a = int(input())
item_od = collections.OrderedDict()
for _ in range(a):
    record_list = re.split(r"(\d+)$", input().strip())
    item_name = record_list[0]
    item_price = int(record_list[1])
    if item_name not in item_od:
        item_od[item_name] = item_price
    else:
        item_od[item_name] = item_od[item_name] + item_price
for i in item_od:
    print(f"{i}{item_od[i]}")

# Word Order
from collections import Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    pass

word_ar = []
a = int(input())
for i in range(a):
    word_ar.append(input().strip())
word_counter = OrderedCounter(word_ar)
print(len(word_counter))
for word in word_counter:
    print(word_counter[word], end=" ")

# Collections.deque()
import collections
a = int(input())
d = collections.deque()
for _ in range(a):
    cmd = list(input().strip().split())
    opt = cmd[0]
    if opt == "append":
        d.append(int(cmd[1]))
    elif opt == "appendleft":
        d.appendleft(int(cmd[1]))
    elif opt == "pop":
        d.pop()
    elif opt == "popleft":
        d.popleft()
for i in d:
    print(i, end=" ")

# Piling Up!
from collections import deque
cas = int(input())
for _ in range(cas):
    a = int(input())
    dq = deque(map(int, input().split()))
    possible = True
    element = (2**31) + 1
    while dq:
        left_element = dq[0]
        right_element = dq[-1]
        if left_element >= right_element and element >= left_element:
            element = dq.popleft()
        elif right_element >= left_element and element >= right_element:
            element = dq.pop()
        else:
            possible = False
            break
    if possible:
        print("Yes")
    else:
        print("No")

# collections.Counter()
n = int(input())
shoe_size = list(map(int, input().split()))
a = int(input())
sell = 0
for _ in range(a):
    s, p = map(int, input().split())
    if s in shoe_size:
        sell = sell + p
        shoe_size.remove(s)
print(sell)

# Time Delta

import datetime
cas = int(input())
time_format = "%a %d %b %Y %H:%M:%S %z"
for _ in range(cas):
    timestamp1 = input().strip()
    
    timestamp2 = input().strip()
    
    time_second1 = datetime.datetime.strptime(timestamp1, time_format)
    
    time_second2 = datetime.datetime.strptime(timestamp2, time_format)
    
    print(int(abs((time_second1 - time_second2).total_seconds())))
    

# Calendar Module
import calendar
import datetime
m, d, y = map(int, input().split())
input_date = datetime.date(y, m, d)
print(calendar.day_name[input_date.weekday()].upper())

# Input()
if __name__ == "__main__":
    x, y = map(int, input().strip().split())
    equation = input().strip()
    print(eval(equation) == y)

# Python Evaluation
eval(input())

# Any or All
n = input()
ar = input().split()
print(all(int(i) > 0 for i in ar) and any(i == i[::-1] for i in ar))

# Exceptions

n = int(input())
for _ in range (n) :
    x, y = input().split()
    try:
        print(int(x) // int(y))
    except Exception as e:
        print(f"Error Code: {e}")

# Zipped!

A, X = map(int, input().split())
scores = []
for _ in range(X):
    scores.append(list(map(float, input().split())))
for i in zip(*scores):
    print(sum(i) / len(i))

# ginortS
s = input()
print(
    "".join(
        sorted(
            s,
            key=lambda x: (
                x.isdigit() and int(x) % 2 == 0,
                x.isdigit(),
                x.isupper(),
                x.islower(),
                x,
            ),
        )
    )
)

# Map and Lambda Function
cube = lambda x: x * x * x

def fibonacci(n):
    ar = [0, 1]
    if n < 2:
        return ar[:n]
    for i in range(2, n):
        ar.append(ar[i - 1] + ar[i - 2])
    return ar



# Re.split()


regex_pattern = r"[.,]+"


# Group(), Groups() & Groupdict()

import re
s = input()
res = re.search(r"([A-Za-z0-9])\1", s)
if res is None:
    print(-1)
else:
    print(res[1])

# Re.findall() & Re.finditer()
import re
s = input()
result = re.findall(
    r"(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([AEIOUaeiou]{2,})(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])",
    s,
)
if result:
    for i in result:
        print(i)
else:
    print(-1)

# Re.start() & Re.end()
import re
s = input().strip()
k = input().strip()
s_len = len(s)
found_flag = False
for i in range(s_len):
    match_result = re.match(k, s[i:])
    if match_result:
        start_index = i + match_result.start()
        end_index = i + match_result.end() - 1
        print((start_index, end_index))
        found_flag = True
if found_flag == False:
    print("(-1, -1)")

# Regex Substitution
import re
import sys

a = int(input())
for line in sys.stdin:
    remove_and = re.sub(r"(?<= )(&&)(?= )", "and", line)
    remove_or = re.sub(r"(?<= )(\|\|)(?= )", "or", remove_and)
    print(remove_or, end="")

# Validating Roman Numerals


regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

# Validating phone numbers
from re import compile, match
a = int(input())
for _ in range(a):
    phone_number = input()
    condition = compile(r"^[7-9]\d{9}$")
    if bool(match(condition, phone_number)):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses
import email.utils
import re
n = int(input())
for _ in range(n):
    s = input()
    parsed_email = email.utils.parseaddr(s)[1].strip()
    match_result = bool(
        re.match(
            r"(^[A-Za-z][A-Za-z0-9\._-]+)@([A-Za-z]+)\.([A-Za-z]{1,3})$", parsed_email
        )
    )
    if match_result:
        print(s)

# Hex Color Code
import re
a = int(input())
for _ in range(a):
    s = input()
    match_result = re.findall(r"(#[0-9A-Fa-f]{3}|#[0-9A-Fa-f]{6})(?:[;,.)]{1})", s)
    for i in match_result:
        if i != "":
            print(i)

# HTML Parser - Part 1
from html.parser import HTMLParser

class CustomHTMLParser(HTMLParser):
    def handle_attr(self, attrs):
        for attr_val_tuple in attrs:
            print("->", attr_val_tuple[0], ">", attr_val_tuple[1])
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        self.handle_attr(attrs)
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        self.handle_attr(attrs)

parser = CustomHTMLParser()
n = int(input())
s = "".join(input() for _ in range(n))
parser.feed(s)
"""
2
<html><head><title>HTML Parser - I</title></head>
<body data-modal-target class='1'><h1>HackerRank</h1><br /></body></html>
"""

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        number_of_line = len(data.split("\n"))
        if number_of_line > 1:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        if data.strip():
            print(data)
    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)
html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser

class CustomHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        self.handle_attrs(attrs)
    def handle_startendtag(self, tag, attrs):
        print(tag)
        self.handle_attrs(attrs)
    def handle_attrs(self, attrs):
        for attrs_pair in attrs:
            print("->", attrs_pair[0].strip(), ">", attrs_pair[1].strip())

n = int(input())
html_string = "".join(input() for _ in range(n))
customHTMLParser = CustomHTMLParser()
customHTMLParser.feed(html_string)
customHTMLParser.close()

# Validating UID
import re
n = int(input())
upper_check = r".*([A-Z].*){2,}"
digit_check = r".*([0-9].*){3,}"
alphanumeric_and_length_check = r"([A-Za-z0-9]){10}$"
repeat_check = r".*(.).*\1"
for _ in range(n):
    uid_string = input().strip()
    upper_check_result = bool(re.match(upper_check, uid_string))
    digit_check_result = bool(re.match(digit_check, uid_string))
    alphanumeric_and_length_check_result = bool(
        re.match(alphanumeric_and_length_check, uid_string)
    )
    repeat_check_result = bool(re.match(repeat_check, uid_string))
    if (
        upper_check_result
        and digit_check_result
        and alphanumeric_and_length_check_result
        and not repeat_check_result
    ):
        print("Valid")
    else:
        print("Invalid")

# Validating Credit Card Numbers
import re
a = int(input())
for _ in range(a):
    credit = input().strip()
    credit_removed_hiphen = credit.replace("-", "")
    valid = True
    length_16 = bool(re.match(r"^[4-6]\d{15}$", credit))
    length_19 = bool(re.match(r"^[4-6]\d{3}-\d{4}-\d{4}-\d{4}$", credit))
    consecutive = bool(re.findall(r"(?=(\d)\1\1\1)", credit_removed_hiphen))
    if length_16 == True or length_19 == True:
        if consecutive == True:
            valid = False
    else:
        valid = False
    if valid:
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes


regex_integer_in_range = r"^[1-9][0-9]{5}$"
regex_alternating_repetitive_digit_pair = r"(?=(\d)\d\1)"



# Matrix Script

import re
n, m = map(int, input().split())
character_ar = [""] * (n * m)
for i in range(n):
    line = input()
    for j in range(m):
        character_ar[i + (j * n)] = line[j]
decoded_str = "".join(character_ar)
final_decoded_str = re.sub(
    r"(?<=[A-Za-z0-9])([ !@#$%&]+)(?=[A-Za-z0-9])", " ", decoded_str
)
print(final_decoded_str)

first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

# Detect Floating Point Number
from re import compile, match
pattern = compile("^[-+]?\d*\.\d+$")
for _ in range(int(input())):
    print(bool(pattern.match(input())))

# XML2 - Find the Maximum Depth


maxdepth = 0
def depth(elem, level):
    global maxdepth
    level = level + 1
    maxdepth = max(maxdepth, level)
    for child in elem:
        depth(child, level)



# XML 1 - Find the Score


def get_attr_number(node):
    total = len(node.attrib.keys())
    for child in node:
        if child:
            total += get_attr_number(child)
        else:
            total += len(child.attrib.keys())
    return total



# Decorators 2 - Name Directory


def person_lister(func):
    def inner(people):
        return [func(p) for p in sorted(people, key=lambda x: (int(x[2])))]
    return inner

    

# Standardize Mobile Number Using Decorators
def wrapper(func):
    def inner(numbers):
        formatted_numbers = ['+91 ' + number[-10:-5] + ' ' + number[-5:] for number in numbers]
        return func(sorted(formatted_numbers)) 
    return inner

# Shape and Reshape
import numpy

ar = list(map(int, input().split()))
np_ar = numpy.array(ar)
print(numpy.reshape(np_ar, (3, 3)))

# Transpose and Flatten
import numpy

n, m = map(int, input().split())
ar = []
for _ in range(n):
    row = list(map(int, input().split()))
    ar.append(row)
np_ar = numpy.array(ar)
print(numpy.transpose(np_ar))
print(np_ar.flatten())

# Concatenate
import numpy

n, m, p = map(int, input().split())
ar1 = []
ar2 = []
for _ in range(n):
    tmp = list(map(int, input().split()))
    ar1.append(tmp)
for _ in range(m):
    tmp = list(map(int, input().split()))
    ar2.append(tmp)
np_ar1 = numpy.array(ar1)
np_ar2 = numpy.array(ar2)
print(numpy.concatenate((np_ar1, np_ar2), axis=0))

# Eye and Identity


import numpy as np
np.set_printoptions(legacy="1.13")
n, m = map(int, input().split())
print(np.eye(n, m, k=0))

# Array Mathematics
import numpy


n, m = map(int, input().split())
ar1 = []
ar2 = []
for _ in range(n):
    tmp = list(map(int, input().split()))
    ar1.append(tmp)
for _ in range(n):
    tmp = list(map(int, input().split()))
    ar2.append(tmp)
np_ar1 = numpy.array(ar1)
np_ar2 = numpy.array(ar2)
print(np_ar1 + np_ar2)
print(np_ar1 - np_ar2)
print(np_ar1 * np_ar2)
print(np_ar1 // np_ar2)
print(np_ar1 % np_ar2)
print(np_ar1**np_ar2)

# Floor, Ceil and Rint
import numpy  as np
np.set_printoptions(legacy="1.13")
A = np.array(input().split(), float)
print(np.floor(A))
print(np.ceil(A))
print(np.rint(A))

# Sum and Prod
import numpy

n, m = map(int, input().split())
ar = []
for _ in range(n):
    tmp = list(map(int, input().split()))
    ar.append(tmp)
np_ar = numpy.array(ar)
s = numpy.sum(np_ar, axis=0)
print(numpy.prod(s))

# Min and Max
import numpy

n, m = map(int, input().split())
ar = []
for _ in range(n):
    tmp = list(map(int, input().split()))
    ar.append(tmp)
np_ar = numpy.array(ar)
print(numpy.max(numpy.min(np_ar, axis=1)))

# Mean, Var, and Std
import numpy

n, m = map(int, input().split())
ar = []
for _ in range(n):
    tmp = list(map(int, input().split()))
    ar.append(tmp)
np_ar = numpy.array(ar)
print(numpy.mean(np_ar, axis=1))
print(numpy.var(np_ar, axis=0))
print(round(numpy.std(np_ar, axis=None), 11))

# Dot and Cross
import numpy

n = int(input())
ar1 = []
ar2 = []
for _ in range(n):
    tmp = list(map(int, input().split()))
    ar1.append(tmp)
np_ar1 = numpy.array(ar1)
for _ in range(n):
    tmp = list(map(int, input().split()))
    ar2.append(tmp)
np_ar2 = numpy.array(ar2)
print(numpy.dot(np_ar1, np_ar2))

# Inner and Outer
import numpy

np_ar1 = numpy.array(list(map(int, input().split())))
np_ar2 = numpy.array(list(map(int, input().split())))
print(numpy.inner(np_ar1, np_ar2))
print(numpy.outer(np_ar1, np_ar2))

# Polynomials
import numpy

p = numpy.array(list(map(float, input().split())), float)
x = float(input())
print(numpy.polyval(p, x))

# Linear Algebra
import numpy as np
np.set_printoptions(legacy="1.13")
n = int(input())
array = np.array([input().split() for _ in range(n)], float)
print(np.linalg.det(array))


# Arrays

import sys
def reverse_array(arr):
    np_array = numpy.array(arr, float)
    return np_array[::-1]  
def arrays(arr):
    return reverse_array(arr)

# Zeros and Ones
import numpy
shape = tuple(map(int, input().split()))
print(numpy.zeros(shape, dtype=int))
print(numpy.ones(shape, dtype=int))

# Birthday Cake Candles
def birthdayCakeCandles(candles):
    max_height = max(candles)
   
    return candles.count(max_height)


import math
import os
import random
import re
import sys


 
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps

def kangaroo(x1, v1, x2, v2):
  
    if v1 == v2:
        return "YES" if x1 == x2 else "NO"
   
    if (x2 - x1) * (v1 - v2) < 0 or (x2 - x1) % (v1 - v2) != 0:
        return "NO"
    else:
        return "YES"

if __name__ == '__main__':
    x1, v1, x2, v2 = map(int, input().split())
    result = kangaroo(x1, v1, x2, v2)
    print(result)

# Viral Advertising

def viralAdvertising(n):
    shared = 5
    cumulative_likes = 0
    
    for day in range(1, n + 1):
        liked = shared // 2
        cumulative_likes += liked
        shared = liked * 3
    return cumulative_likes

if __name__ == '__main__':
    n = int(input().strip())
    result = viralAdvertising(n)
    print(result)

# Recursive Digit Sum
def superDigit(n, k):
    
    initial_sum = sum(int(digit) for digit in n) * k
    
  
    def find_super_digit(x):
        if x < 10:
            return x
        return find_super_digit(sum(int(digit) for digit in str(x)))
    
    return find_super_digit(initial_sum)

if __name__ == '__main__':
    n, k = input().split()
    k = int(k)
    result = superDigit(n, k)
    print(result)

# Insertion Sort - Part 1

def insertionSort1(n, arr):
    key = arr[-1] 
    i = n - 2
    while i >= 0 and arr[i] > key:
        arr[i + 1] = arr[i]  
        print(*arr)  
        i -= 1
    arr[i + 1] = key  
    print(*arr)  
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
def insertionSort2(n, arr):
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        print(*arr)
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

# Company Logo
from collections import Counter
def most_common_characters(s):
    counter = Counter(s)
    sorted_characters = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    for char, freq in sorted_characters[:3]:
        print(char, freq)
if __name__ == '__main__':
    s = input().strip()
    most_common_characters(s)


# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    arr.sort(key=lambda row: row[k])
    for row in arr:
        print(*row)

