library(sparklyr)
library(dplyr)
library(sparkapi)

sc <- spark_connect(master = "local")

absences <- read.csv("https://raw.githubusercontent.com/IBMPredictiveAnalytics/Gradient_Boosted_Trees_with_MLlib/master/example/student-mat.csv", sep = ";" )

#codebook
codebook <- "https://archive.ics.uci.edu/ml/datasets/Student+Performance"



# copy to spark
absences_tbl <- copy_to(sc, absences, overwrite = T)

# I want to predict G3 based on pstatus, school, sex, age, studytime, & absences
# pstatus, school, & sex are strings. need to convert to binary

# using basic dplyr
absences_2_tbl <- absences_tbl %>% mutate(p_live_together = ifelse(absences == "T", 1, 0),
                                       male = ifelse(sex == "M", 1, 0),
                                       school_GP = ifelse(school == "GP", 1, 0)))

# create the index / binary columns 
# sdf_mutate = spark data frame mutate allows spark functions 
## ft_string_indexer creates dummy variable
  # going to check this using basic dplyr
absences_tbl <- absences_tbl %>% sdf_mutate(p_live_together = ft_string_indexer("Pstatus"), 
                                            school_index = ft_string_indexer("school"),
                                            sex_index = ft_string_indexer("sex"))

# create gradient boosted trees
absences_tbl %>%  ml_gradient_boosted_trees(response = "G3",
                          features = c("p_live_together", "sex_index", "school_index", "studytime", "absences"))

# using sparklyr to partition data
partition_tbl <- sdf_partition(absences_tbl, training = 0.75, test = 0.25, seed = 0)

# create gradient boosted tree model
gbt <- partition_tbl$training %>% ml_gradient_boosted_trees(response = "G3",
                         features = c( "p_live_together", "sex_index", "school_index", "studytime", "absences"))

gbt_predict <- sdf_predict(gbt, newdata = partition_tbl$test) %>% collect

#test
gbt_predict$prediction

test_g3 <- partition_tbl$test %>% select(G3)

                              
table(gbt_predict$prediction, test_g3)
