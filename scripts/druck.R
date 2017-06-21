library(readr)
library(ggplot2)
library(viridis)

# ------------------------------ SETTINGS --------------------------------------------------

setwd("~/ProKlaue/testdaten/druck")
ground_type = "gummi"
bone = "Klaue 1 (K1T)"


# ------------------------------ TRANSFORM FIT --------------------------------------------------

trans_func <- function(x, mat=transform){return((mat %*% matrix(c(x["x"], x["y"], 1),nrow=3))[1:2,])}

rotation_matrix <- function(alpha, radian=TRUE)
{
  if(!radian)
    alpha = (alpha %% 360) *pi/180
  return(matrix(c(cos(alpha),sin(alpha),-sin(alpha),cos(alpha)), nrow=2, ncol=2))
}

rot = rotation_matrix(4, radian=F)
rot_pivot = matrix(c(7.4,9), nrow=2)
displace = matrix(c(0,0.3), nrow=2)

transform_extra = rbind(cbind(c(1,0),c(0,1),displace),c(0,0,1))%*%
  rbind(cbind(c(1,0),c(0,1),rot_pivot),c(0,0,1))%*%
  rbind(cbind(rot,c(0,0)),c(0,0,1))%*%
  rbind(cbind(c(1,0),c(0,1),-rot_pivot),c(0,0,1))


transform_imprint = unname(as.matrix(read.csv(paste0(bone, "/transformation_imprint_",ground_type, ".csv"), header = F)))

transform = transform_extra%*%transform_imprint

pressure_data <- read.csv(paste0(bone, "/pressure_data_",ground_type, ".csv"))
pressure_data$pressure[pressure_data$pressure==0]=NA

segments_imprint_right_orig <- read_csv(paste0(bone, "/segments_imprint_right", ".csv"))
segments_imprint_left_orig <- read_csv(paste0(bone, "/segments_imprint_left", ".csv"))

segments_measurements_right_orig <- read_csv(paste0("segments_for_measurements_right", ".csv"))
segments_measurements_left_orig <- read_csv(paste0("segments_for_measurements_left", ".csv"))

transform_zones_left = unname(as.matrix(read.csv(paste0(bone, "/transformation_zones_left_",ground_type, ".csv"), header = F)))
transform_zones_right = unname(as.matrix(read.csv(paste0(bone, "/transformation_zones_right_",ground_type, ".csv"), header = F)))

transform_zones_left = transform_extra%*%transform_zones_left 
transform_zones_right = transform_extra%*%transform_zones_right


segments_imprint_right <- segments_imprint_right_orig
segments_imprint_left <- segments_imprint_left_orig
segments_measurements_right <- segments_measurements_right_orig 
segments_measurements_left <- segments_measurements_left_orig

segments_imprint_left[c("x", "y")] = t(apply(segments_imprint_left, 1,trans_func, mat = transform))
segments_imprint_right[c("x", "y")] = t(apply(segments_imprint_right, 1, trans_func, mat = transform))

segments_measurements_left[c("x", "y")] = t(apply(segments_measurements_left_orig, 1, trans_func, mat = transform_zones_left))
segments_measurements_right[c("x", "y")] = t(apply(segments_measurements_right_orig, 1, trans_func, mat = transform_zones_right))



# ----------------------------------- PLOTS -------------------------------------------------------------

# plot with imprint
ggplot()+
  geom_vline(xintercept = -10:40, alpha = 0.3)+
  geom_hline(yintercept = -10:40, alpha = 0.3)+
  geom_vline(xintercept = (-100:400)/10, alpha = 0.1)+
  geom_hline(yintercept = (-100:400)/10, alpha = 0.1)+
  geom_raster(data = pressure_data, aes(x=x, y=y, fill=pressure))+
  geom_polygon(data = segments_imprint_left, aes(x=x, y=y, group=SID), fill=NA, color="black")+
  geom_polygon(data = segments_imprint_right, aes(x=x, y=y, group=SID), fill=NA, color="black")+
  scale_fill_viridis(option = "viridis", na.value="NA")+
  scale_x_continuous(limits = c(0,18))+
  scale_y_continuous(limits = c(0,18)) +
  coord_fixed()


# plot with imprint and zones
ggplot()+
  geom_vline(xintercept = -10:40, alpha = 0.3)+
  geom_hline(yintercept = -10:40, alpha = 0.3)+
  geom_vline(xintercept = (-100:400)/10, alpha = 0.1)+
  geom_hline(yintercept = (-100:400)/10, alpha = 0.1)+
  geom_raster(data = pressure_data, aes(x=x, y=y, fill=pressure))+
  geom_polygon(data = segments_imprint_left, aes(x=x, y=y, group=SID), fill=NA, color="black")+
  geom_polygon(data = segments_imprint_right, aes(x=x, y=y, group=SID), fill=NA, color="black")+
  geom_polygon(data = segments_measurements_left, aes(x=x, y=y, group=SID), fill=NA, color="black")+
  geom_polygon(data = segments_measurements_right, aes(x=x, y=y, group=SID), fill=NA, color="black")+
  scale_fill_viridis(option = "viridis", na.value="NA")+
  scale_x_continuous(limits = c(0,16))+
  scale_y_continuous(limits = c(0,16)) +
  coord_fixed()


# ------------------------------ SAVE TRANSFORMED FIT ----------------------------------------
write.table(file = paste0(bone, "/transformation_r_imprint_",ground_type, ".csv"), x= transform, row.names = FALSE, col.names=FALSE, sep=",")
write.table(file = paste0(bone, "/transformation_r_zones_left_",ground_type, ".csv"), x= transform_zones_left, row.names = FALSE, col.names=FALSE, sep=",")
write.table(file = paste0(bone, "/transformation_r_zones_right_",ground_type, ".csv"), x= transform_zones_right, row.names = FALSE, col.names=FALSE, sep=",")


# ------------------------------ READ TRANSFORMATION ----------------------------------------
transform = unname(as.matrix(read.csv(file = paste0(bone, "/transformation_r_imprint_",ground_type, ".csv"), header=F)))
transform_zones_left = unname(as.matrix(read.csv(paste0(bone, "/transformation_r_zones_left_",ground_type, ".csv"), header = F)))
transform_zones_right = unname(as.matrix(read.csv(paste0(bone, "/transformation_r_zones_right_",ground_type, ".csv"), header = F)))

segments_imprint_right <- segments_imprint_right_orig
segments_imprint_left <- segments_imprint_left_orig
segments_measurements <- segments_measurements_right_orig 
segments_measurements <- segments_measurements_left_orig

segments_imprint_left[c("x", "y")] = t(apply(segments_imprint_left, 1,trans_func, mat = transform))
segments_imprint_right[c("x", "y")] = t(apply(segments_imprint_right, 1, trans_func, mat = transform))

segments_measurements_left[c("x", "y")] = t(apply(segments_measurements_left_orig, 1, trans_func, mat = transform_zones_left))
segments_measurements_right[c("x", "y")] = t(apply(segments_measurements_right_orig, 1, trans_func, mat = transform_zones_right))

# ------------------------------ STATISTICS --------------------------------------------------
statistics = read.csv(paste0(bone, "/statistics_",ground_type, ".csv"))

# ------------------------------ STATISTIC PLOTS --------------------------------------------
ggplot(statistics, aes(group = side, x=SID, y=pressure_rel_to_area, fill=side))+geom_col(position = "dodge")

