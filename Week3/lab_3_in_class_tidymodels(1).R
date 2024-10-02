library(tidymodels)

data <- read.csv("lab_3_data.csv", header = TRUE)

# create a split object that holds the training and testing sets using the initial_split function
# the arguments to the initial_split function are the data to be split, the variable to stratify on,
# and the proportion of the data that will be in the training set
data_split <- initial_split(data, strata = "lodgepole_pine", prop = 0.75)

# use the training and testing functions on the split object created above to extract the training
# and testing sets
training_set <- training(data_split)
test_set  <- testing(data_split)

# specify the recipe that will be used to pre-process any data set that has the structure specified in
# the recipe function
pine_recipe <- 
# the main argument to the recipe function is an R formula that specifies the dependent variable (target) on
# the left-hand side and the independent variables (features) on the right-hand side
# We also need to pass the recipe function some data, so it can learn the structure of the data it will
# need to process. We can use the recipe to pre-process any dataset that looks like the data specified (given
# by training_set in this particular case below)
  recipe(
    lodgepole_pine ~ elevation + aspect + slope + horizontal_distance_to_hydrology +
      vertical_distance_to_hydrology + horizontal_distance_to_roadways + hillshade_9am + hillshade_noon + 
      hillshade_3pm + horizontal_distance_to_fire_points + wilderness_area + soil_type, 
    data = training_set
  ) %>%
  step_other(soil_type) %>% #step_other creates an "other" category for the categorical variable(s) specified. Categories that appear less than the threshold amount specified (or the default amount if no threshold is specified) are grouped into the "other" category
  step_dummy(all_nominal_predictors()) %>% #step_dummy performs one-hot encoding
  step_center(all_predictors()) %>% #step_center centers the variables specified: the resulting centered variables will have a mean of zero
  step_scale(all_predictors()) %>% #step_scale scales the variables specified: the resulting scaled variables will have a standard deviation of one
  prep(training = training_set, retain = TRUE) #prep prepares the recipe, meaning that the recipe will learn any values needed to perform the pre-processing steps on a dataset. For example, the recipe needs to learn the means of the variables, so that step_center can be applied. The "training" argument specifies the dataset on which this learning will take place

training_set_baked <- bake(pine_recipe, new_data = training_set) #apply the recipe to the training set
test_set_baked <- bake(pine_recipe, new_data = test_set) #apply the recipe to the testing set


######################### the rest is the same as in lab_3_in_class.R #########################
# convert to R arrays: these are now ready to be used in Keras functions
training_features <- array(data = unlist(training_set_baked[, -11]),
                           dim = c(nrow(training_set_baked), 20))
training_labels <- array(data = unlist(training_set_baked[, 11]),
                         dim = c(nrow(training_set_baked)))

test_features <- array(data = unlist(test_set_baked[, -11]),
                       dim = c(nrow(test_set_baked), 20))
test_labels <- array(data = unlist(test_set_baked[, 11]),
                     dim = c(nrow(test_set_baked)))

