### Template to create visualisation.

# [1] Create the ggplot object (Note: the data and mapping objects are on 
# the first line in the layer function and "Diamonds" is the name of the data):
ggplot() +
  layer( 
    data = diamonds, mapping = aes(x - carat, y = price),  # [1a] Set data and mapping objects.
    geom = "point", stat = "identity", position = "identity"
  ) +  # [1b] Set geom type, stat, and position.
  scale_y_continuous() +  # [1c] Add scale function for y.
  scale_x_continuous() +  # [1d] Add scale function for x.
  coord_cartesian()  # [1e] Set the coordinates

### Simplify the code

ggplot() +  # [1] Create the object.
  layer(data = diamonds, mapping = aes(x = carat, y = price),  
        # [2] Specify the data set and the x-axis and y-axis.
        geom = "point"  
        # [3] Set the geom (with defaults for stat, position, scale, and coords).
        )

### Simplify further

# [1] Place the calls for data and axes in the ggplot function 
# (thereby removing the need for the layer function):

ggplot(diamonds, aes(carat, price)) + 
  geom_point() # Note: this object’s geom layer function.

ggplot(diamonds, aes(carat, price)) +
  geom_point() +
  geom_smooth()  # [1] Add the smoothed conditional mean of y given x.
