### Practical Activity 4.1.8
# Create a vector/list representing each column:
Name <- c("Donna", "Anthea", "Yusuf", "Bongile", "Abe", "Quentin", "Tracy",
           "Bilal", "Victoria", "Indira")

Last_Name <- c("Watusi", "Smits", "Zayeed", "Maleson", "Dawidowitz", "Ng",
                "Jacks", "Ghani", "LeGrande", "De Silva")

Gender <- c("F", "F", "M", "F", "M", "N", "F", "M", "F", "F")

Age <- c(38, 29, 44, 24, 66, 34, 55, 40, 38, 29)

Email <- c("donnaw@gotmail.co.uk", "as@workmail.co.nz", "yusufzz@glammail.com",
           "bongi1@workmail.co.uk", "papabe@oldmail.com", "qng7@mailio.com",
           "tracy@blurredv.co.uk", "bghani2@mailio.com", "beachh@glammail.com",
           "indigirl@workmail.com")

Purchases <- c(24, 16, 28, 33, 21, 30, 28, 21, 22, 32)

# Combine six vectors into a data frame.
book_df <- data.frame(Name, Last_Name, Gender, Age, Email, Purchases)

# Print data frame.
book_df

# Check the data frame.
typeof(book_df)
class(book_df)
dim(book_df)

# Change the column names to upper-case.
names(book_df) <- c("NAME", "LAST NAME", "GENDER", "AGE", "EMAIL",
                    "PURCHASES")

print(book_df)

# Add a new column displaying "ID".
ID <- c(2101:2110)

# Add the new column to the current data frame.
book_df$ID <- ID

# Print the data frame.
print(book_df)

# Return the structure of the data frame.
str(book_df)

# Subset the data frame.
# Extract only female customers.
subset(book_df, Gender == "F")
# Extract all customers over the age of 50
subset(book_df, AGE > 50)
# Extract all customers with less than 20 purchases in 2021
subset(book_df, PURCHASES < 20)
