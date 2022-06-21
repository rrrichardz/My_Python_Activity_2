# First code.
print("Hello world")

# Creating an object.
# Assign 'a' the numeric value '10' using the 'assigned to' operator, creating an object:
a <- 10

# Print the object.
a

x <- 120
y <- "goldfish"
z <- 12.25
c <- "4x"
v <- "Millennium Falcon"

x
y
z
c
v

# Check the class of the object ('a' has been assigned the number 10):
class(a)

# Check the type of object.
typeof(a)

# Check the length of an object.
length(a)

# Check object attributes.
attributes(a)

name <- "Andrew" # [1] Assign the object name.

typeof(name) # [2] Specify the typeof() function.

length(name) # [3] Specify the length() function.

x <- c(1, 2, 3) # [1] Create a vector and assign it to the object x.
attributes(x)

named.vec <- c("First" = 1, "Second" = 2, "Third" = 3) # [1] Create an object and [2] a vector.
attributes(named.vec) # [3] Find the attributes of named.vec.

c(1, 2, 3, 4, 5, 6, 7) # Create a numeric vector.

# [1] Create a vector, specify the assign() function, and [2] call the c() function:
assign("vector1", c(1, 2, 3, 4, 5, 6, 7))

vector1 # [3] Print the vector.

length(vector1)

# Determine the vector type.
str(1, 5.5, 100)

# Use the '/' operator to divide zero by zero.
0 / 0

# Check the vector type of the object 1L.
str(1L)

# [1] Assign to x the integers in the brackets.
x <- c(4L, 6L)

# [2] Check whether x is an integer.
is.integer(x)

# Create the vector and assign it to 'vec'.
vec <- c(FALSE, NA, TRUE, FALSE)

# Return the elements of the object 'vec'.
vec

# [1] Create the vector, specify the c() function and
# pass the characters; [2] Assign the vector to object authors:
authors <- c("Andrew", "Joy")

# [3] Check the vector type.
str(authors)

# [1] Specify/create a complex vector and [2] assign
# it to the object 'comp':
comp <- c(2-4i, 1+3i-2, 55i)

# [3] Run the typeof() function on the object 'comp'
# and place it inside the print() function so that
# the type in the output is returned:
print(typeof(comp))

# Recognise the assign operator, where the object x
# has been assigned the value 10:
x <-10
y <- 5 # Assign y the value 5.

x + y # Add the object x to y

# [1] Assign values to the object 'b'.
b <- 10:15

# [2] Print out the value of b.
b

# Look for the value of index 1.
b [1]

# Look for the value of index 5.
b [5]

# [1] Create a vector using the c() function and
# assign it to the object x:
x <- c(1, 2, 3, 4, 5)

# [2] Create a vector using the c() function and
# assign it to the object y:
y <- c("yes", "no")

# [3] Find the types of each vector:
str(x)
str(y)

# [1] Coerce vector x to character.
as.character(x)

# [2] Coerce vector y to logical.
as.logical(y)

X <- c(TRUE, "False")
X <- c(1.2, 99i)
X <- c("One", 1)

