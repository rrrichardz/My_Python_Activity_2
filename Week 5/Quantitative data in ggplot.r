### Quantitative data

## 1. Change the geom layer

# Specify the ggplot function:
ggplot(new_wages, aes(x = age)) +
  geom_density() + # Specify the geom_density.
  labs(title = "Participants by age") # Specify title.

## 2. Add colour

# Specify the ggplot function:
ggplot(new_wages, aes(x = age)) +
  geom_density(fill = "red",
               bw = 1) + # Add fill colour to the function.
  labs(title = "Participants by age") # Specify title.

# Specify the ggplot function:
ggplot(new_wages, aes(x = age)) +
  geom_density(fill = "red",
               bw = 0.5) + # Add fill colour to the function.
  labs(title = "Participants by age",
       subtitle = "Bandwidth = 0.5",
       x = ("Age"),
       y = ("Density")) + # Specify title and labels.
  theme_minimal()
