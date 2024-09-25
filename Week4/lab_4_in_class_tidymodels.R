library(tidymodels)

data <- read.csv("lab_4_data.csv", header = TRUE)

data_split <- initial_split(data, strata = "lodgepole_pine", prop = 0.75)

training_set <- training(data_split)
test_set  <- testing(data_split)

pine_recipe <- 
  recipe(
    lodgepole_pine ~ elevation + aspect + slope + horizontal_distance_to_hydrology +
      vertical_distance_to_hydrology + horizontal_distance_to_roadways + hillshade_9am + hillshade_noon + 
      hillshade_3pm + horizontal_distance_to_fire_points + wilderness_area + soil_type, 
    data = training_set
  ) %>%
  step_other(soil_type) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep(training = training_set, retain = TRUE)

training_set_baked <- bake(pine_recipe, new_data = training_set)
test_set_baked <- bake(pine_recipe, new_data = test_set)

training_features <- array(data = unlist(training_set_baked[, -11]),
                           dim = c(nrow(training_set_baked), ncol(training_set_baked)-1))
training_labels <- array(data = unlist(training_set_baked[, 11]),
                         dim = c(nrow(training_set_baked)))

test_features <- array(data = unlist(test_set_baked[, -11]),
                       dim = c(nrow(test_set_baked), ncol(training_set_baked)-1))
test_labels <- array(data = unlist(test_set_baked[, 11]),
                     dim = c(nrow(test_set_baked)))

##### training dense feed-forward neural networks in Keras #####

library(reticulate)
library(tensorflow)
library(keras)

use_virtualenv("my_tf_workspace")

model <- keras_model_sequential() %>%
  layer_dense(units = 20, activation = "relu") %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model

compile(model,
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = "accuracy")

random_shuffle <- sample(1:nrow(training_features))
training_features <- training_features[random_shuffle, ]
training_labels <- training_labels[random_shuffle]
history <- fit(model, training_features, training_labels,
               epochs = 250, batch_size = 512, validation_split = 0.33)

plot(history)

predictions <- predict(model, test_features)
head(predictions, 10)

predicted_class <- (predictions[, 1] >= 0.5) * 1
head(predicted_class, 10)
