### Creating a matrix
# [1] Create a matrix; [2a] specify a sequence of numerals,
# [2b] define the number of rows, and [2c] define the number
# of columns:
B <- matrix (1:9, nrow = 3, ncol = 3)
print (B)

# [1] Create a matrix B.
B <- matrix (1:9, nrow = 3, ncol = 3)

# [2] Transpose the data in the matrix B.
t (B)

### Dimensions and attributes of a matrix
dim (B)
attributes (B)

### Combining two matrices
# Create matrix A.
A = matrix (1:6, nrow = 3, ncol = 2)
# Create matrix B.
B = matrix (1:9, nrow = 3, ncol = 3)
# Combine matrix A and B.
cbind (A, B)

# Create a matrix C.
C = matrix (1:6, nrow = 2, ncol = 3)
# Create a matrix D.
D = matrix (1:9, nrow = 3, ncol = 3)
# Combine matrix C and D.
rbind (C, D)

### Extracting elements from a matrix
# [1] Create a matrix with [1a] the numerals 1-9,
# [1b] 3 rows, [1c] 3 columns, and [1d] "byrow = True":
Z <- matrix (1:9, nrow = 3, ncol = 3, byrow = TRUE)
# [2] Print the matrix.
print (Z)

# Extract the element in the 2nd row and 2nd column:
Z [2, 2]

# Extract two columns from Z ("1" and "3" indicates
# the 1st and 3rd columns for extraction):
Z [, c(1, 3)]
# Extract two rows from Z.
Z [c(1, 3), ]

### Naming rows and columns
# [1] Create/specify the rows.
rownames (Z) <- c("Top", "Middle", "Bottom")
# [2] Create/specify the columns.
colnames (Z) <- c("Left", "Middle", "Right")
# [3] Print the matrix Z.
print (Z)
