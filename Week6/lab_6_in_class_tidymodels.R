library(tidymodels)

data <- read.csv("lab_6_data.csv", header = TRUE)

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
  layer_dense(units = 70, activation = "relu") %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model

# methods for regularization to prevent overfitting are:
# stop training early (choose number of epochs using learning curves)
# kernel_regularizer = regularizer_l2(l = 0.01)
# layer_dropout(rate = 0.6)
# https://medium.com/analytics-vidhya/a-simple-introduction-to-dropout-regularization-with-code-5279489dda1e

compile(model,
        optimizer = optimizer_rmsprop(learning_rate = 0.01),
        loss = "binary_crossentropy",
        metrics = "accuracy")

random_shuffle <- sample(1:nrow(training_features))
training_features <- training_features[random_shuffle, ]
training_labels <- training_labels[random_shuffle]
history <- fit(model, training_features, training_labels,
               epochs = 40, batch_size = 512, validation_split = 0.33)

plot(history)

predictions <- predict(model, test_features)

test_results <-
  test_set %>%
  select(lodgepole_pine) %>%
  bind_cols(
    data.frame(p_1 = predictions)
  )

################################### plot ROC curve ###################################

roc_data <- data.frame(threshold=seq(1,0,-0.01), fpr=0, tpr=0)
for (i in roc_data$threshold) {
  
  over_threshold <- test_results[test_results$p_1 >= i, ]
  
  fpr <- sum(over_threshold$lodgepole_pine==0)/sum(test_results$lodgepole_pine==0)
  roc_data[roc_data$threshold==i, "fpr"] <- fpr
  
  tpr <- sum(over_threshold$lodgepole_pine==1)/sum(test_results$lodgepole_pine==1)
  roc_data[roc_data$threshold==i, "tpr"] <- tpr
  
}

ggplot() +
  geom_line(data = roc_data, aes(x = fpr, y = tpr, color = threshold), linewidth = 2) +
  scale_color_gradientn(colors = rainbow(3)) +
  geom_abline(intercept = 0, slope = 1, lty = 2) +
  geom_point(data = roc_data[seq(1, 101, 10), ], aes(x = fpr, y = tpr)) +
  geom_text(data = roc_data[seq(1, 101, 10), ],
            aes(x = fpr, y = tpr, label = threshold, hjust = 1.2, vjust = -0.2))


################################### ROC curve calculation breakdown ###################################

ggplot(data = test_results, aes(x = p_1, y = lodgepole_pine)) +
  geom_jitter()

threshold <- 0.6

test_results$predictions <- ifelse(test_results$p_1 >= threshold, 1, 0)
tp <- nrow(test_results[test_results$lodgepole_pine==1 & test_results$predictions==1, ])
fp <- nrow(test_results[test_results$lodgepole_pine==0 & test_results$predictions==1, ])
tn <- nrow(test_results[test_results$lodgepole_pine==0 & test_results$predictions==0, ])
fn <- nrow(test_results[test_results$lodgepole_pine==1 & test_results$predictions==0, ])

test_results$type <- ""
test_results[test_results$lodgepole_pine==1 & test_results$predictions==1, "type"] <- "tp"
test_results[test_results$lodgepole_pine==0 & test_results$predictions==1, "type"] <- "fp"
test_results[test_results$lodgepole_pine==0 & test_results$predictions==0, "type"] <- "tn"
test_results[test_results$lodgepole_pine==1 & test_results$predictions==0, "type"] <- "fn"

ggplot(data = test_results, aes(x = p_1, y = lodgepole_pine)) +
  geom_jitter(aes(colour = type)) +
  geom_vline(xintercept = threshold, linetype = "dashed", color = "blue", linewidth = 1.5) +
  scale_color_brewer(palette = "RdYlBu")

fpr <- fp/(fp + tn)
tpr <- tp/(tp + fn)


