concrete <- read.csv('800-concrete/concrete.csv', stringsAsFactors = F)[-1]
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

concrete <- as.data.frame(lapply(concrete, normalize))

max_index <- length(concrete$strength)
slice_index <- round(max_index * 0.75)
training_data <- concrete[1:slice_index,]
test_data <- concrete[slice_index:max_index,]

library(neuralnet)
concrete_model <-
  neuralnet(strength ~ cement + flag + ash + water + superplastic + coarseagg +
              findagg + age, data = training_data, hidden=3)
# plot(concrete_model) # plot the neural networks
model_results <- compute(concrete_model, test_data[1:8])

cor(model_results$net.result, test_data$strength)
plot(concrete_model)
