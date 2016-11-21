#install.packages("devtools")
devtools::install_github("rstudio/sparklyr")

# load sparklyr & install spark
library(sparklyr)
spark_install(version = "1.6.1")

# load dplyr and create spark connection
library(dplyr)
sc <- spark_connect(master = "local")

# Loading local data
wine <- read.csv("/Users/Josiah/Documents/Dataa/Data Sets/wine_classification.csv")
wine_tbl <- copy_to(sc, wine)

set.seed(0)

# Create initial model
fit <- wine_tbl %>% ml_linear_regression(response = "quality",
                            features = c("pH", "alcohol", "density", "type"))

# Note that spark doesn't like strings. I'm converting the quality to a dummy variable where white = 1
wine_tbl <- wine_tbl %>% mutate(white = ifelse(type == "White", 1, 0))


# Creating model using sparklyr with new dummy variable
fit <- wine_tbl %>% ml_linear_regression(response = "quality",
                                         features = c("pH", "alcohol", "density", "white"))

# creating lm using base functions 
fit_base <- lm(quality ~ pH + alcohol + density + white, data = wine_tbl)

#compare models
summary(fit)
summary(fit_base)

# Since I can't use the spark tbl I will create the same data as a regular dplyr table
wine <- wine %>% mutate(white = ifelse(type == "White", 1, 0))

# Create k-means using base
base_kmeans <- kmeans(wine[, c("quality", "pH", "alcohol", "density", "white")], 3, iter.max = 10)

# Create k-means using spark
spark_kmeans <-  wine_tbl %>% ml_kmeans(centers = 3, iter.max = 10,
                                 features = c("quality", "pH", "alcohol", "density", "white"))

# Time to compare the centers 
# creating data frame from kmeans centers
base_centers <- data.frame(base_kmeans$centers)

# Printing centers of base and spark
arrange(base_centers, quality)
arrange(spark_kmeans$centers, quality)

# Notice how the outputs are very similar. 

# Going to test the logistic regression capabilities 

# load the data 
loan_data <- readRDS("/Users/Josiah/Documents/Dataa/Data Sets/loan_data.rds")
loan_data_tbl <- copy_to(sc, loan_data) # copying it to spark

base_logit_loan <- glm(loan_status ~ loan_amnt + age + emp_length, family = binomial, data = loan_data_tbl)

spark_logit_loan <- loan_data_tbl %>% ml_logistic_regression(response = "loan_status",
                                                            features = c("loan_amnt", "age", "emp_length"))
# Can't incorporate "emp_length", throws something about NA values
# Note that spark aborts immediately due to null values

# attempt to use annual_inc, it is a double not int. Is there a way to change column types
spark_logit_ann_inc <- ml_logistic_regression(loan_data_tbl, response = "loan_status", 
                                              features = c("loan_amnt","age", "annual_inc"))
  # this creates a summary of just coefficients, rocl, auc. I want to be able to view p-val, std,

# Recreate glm() using same predictors as spark logit
base_logit_ann_inc <- glm(loan_status ~ loan_amnt + age + annual_inc, family = binomial, data = loan_data_tbl)

# see summary both
summary(base_logit_ann_inc)
summary(spark_logit_ann_inc)

# almost identical just different due to rounding