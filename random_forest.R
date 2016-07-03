library(sparklyr)
library(dplyr)
library(sparkapi)

sc <- spark_connect(master = "local")

absences <- read.csv("https://raw.githubusercontent.com/IBMPredictiveAnalytics/Gradient_Boosted_Trees_with_MLlib/master/example/student-mat.csv", sep = ";" )

#codebook
codebook <- "https://archive.ics.uci.edu/ml/datasets/Student+Performance"


# copy to spark
absences_tbl <- copy_to(sc, absences, overwrite = T)

# I want to predict average score  based on pstatus, school, sex, age, studytime, & absences
# pstatus, school, & sex are strings. need to convert to binary

absences_tbl <- absences_tbl %>% mutate(avg_score = (G1 + G2 + G3)/3)

# create binary
absences_tbl <-  absences_tbl %>% sdf_mutate(p_live_together = ft_string_indexer("Pstatus"), 
                                             school_index = ft_string_indexer("school"),
                                             sex_index = ft_string_indexer("sex")) 


# using sparklyr to partition data
partition_tbl <- sdf_partition(absences_tbl, training = 0.75, test = 0.25, seed = 0)


# Create random forest
rf <- partition_tbl$test %>% 
  ml_random_forest(response = "avg_score", #apparently I can't use a period in variable name
                   features = c("school_index", "sex_index", "age", "studytime", "health", "p_live_together"))

rf_predict <- sdf_predict(rf, newdata = partition_tbl$test) %>% collect
rf_stuff <- merge(rf_predict$avg_score, rf_predict$prediction)

sqrt(mean((rf_stuff$x - rf_stuff$y)^2))
