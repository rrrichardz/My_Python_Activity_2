### Creating lists
# [1] Create an object 'list_data' and [2] a list using
# the list() function:
list_data <- list("Red", "Green", "Yellow")

print(list_data)

# Create a list and assign it to the object 'basket_apples':
basket_apples  <-  list ("Red", "Green", "Yellow") 

print (basket_apples)  # Return the objects in the new list.

### Adding list elements
# [1] Create the list.
basket_apples <- list ("Red", "Green", "Yellow")

# [2] Add an element 'White' to the 4th index in the list:
basket_apples [4] <- "White"

print(basket_apples)

# Add elements to the list:
basket_apples [1] <- "Granny Smith"
basket_apples [2] <- "Citrus"
basket_apples [3] <- "Flower"

print(basket_apples)

### Removing list elements
# Assign the 4th element in the index the value NULL.
basket_apples [4] <- NULL

print (basket_apples)

# Replace the 3rd element.
basket_apples [3] <- "Pink"

print (basket_apples)

### Merging lists
# [1] Create a list with 5, 6 and 7.
list1 <- list (5, 6, 7)
# [2] Create a second list with "Thu", "Fri" and "Sat".
list2 <- list ("Thu", "Fri", "Sat")
# [3] Append the lists together.
merged_list <- c (list1, list2)
# [4] Print the new list.
print (merged_list)

### Indexing lists
# Create a list named days.week:
days_week <- list ("Sunday", "Monday", "Tuesday", 
                   "Wednesday", "Thursday", "Friday", 
                   "Saturday")

# Call the element in the 5th index.
days_week [5]

# Create a list "days_week":
days_week  <-  list ("Sunday", "Monday", "Tuesday", 
                     "Wednesday", "Thursday", "Friday", 
                     "Saturday") 

# Call the element in the 3rd index (double barckets):
days_week [[3]]
