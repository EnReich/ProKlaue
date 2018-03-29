# This script reads in data produced by the maya angle script. 
# The data is processed (axes names and orientations, time scale, averages) 
# and written into other output formats.
# Finally some plots are produced

library(ggplot2)
library(data.table)
library(viridis)
library(ggthemes)
library(Cairo)
library(Rcpp)
library(inline)


# set the working directory to the folder with the results from the angle script
setwd("~/ProKlaue - Moni/ergebnisse")


# for rolling min
# set the path to the rollingMin Rcpp 
sourceCpp("~/ProKlaue/scripts/rollingMin.cpp")


# specify a folder for the plots (should exist)
plots_dir = "plots"

#specify a folder for the output tables (should exist)
tables_dir = "csv" 

# --------------- VARIABLES TO FIT -------------------------------------
# variables to fit and calculate gradient on the fit
fitting_vars = c("R1", "R2", "RF", "distance")
span_for_fit = 0.2

timestep_for_int = 0.001
xout = seq(-0.01, 1.01, timestep_for_int)


# variables to collect in the data
data_vars = c("R1", "R1_fitted", "R1_grad", 
              "R2", "R2_fitted", "R2_grad",
              "RF", "RF_fitted", "RF_grad", 
              "distance")

# variables to appear in the plot 
plot_vars = c("R1", "R2", "RF") #, "RXZ_fitted", "RXZ_grad", "RXY_fitted", "RXY_grad", "RYZ_fitted", "RYZ_grad", "distance")
plot_vars_2 = c("R1_fitted", "R1_grad", "R2_fitted", "R2_grad", "RF_fitted", "RF_grad", "distance")
plot_vars_3 = c("R1_fitted", "R2_fitted", "RF_fitted", "distance")


# --------------- LABELS -------------------------------------
plot_vars_4 = c("R1_fitted_inv", "R2_fitted", "RF_fitted")
plot_vars_4_raw = c("R1_inv", "R2", "RF")
animal_breaks = c("alma", "berta")
animal_labels = c("Alma", "Berta")
ground_breaks = c("beton", "karera", "kura", "pedikura" )
ground_labels = c("beton" = "Beton", "kura" = "Kura", "pedikura" = "Pedikura", "karera" = "Karera") #c("Beton", "Kura", "Pedikura", "Karera")
ground_labeler = as_labeller(c("beton" = "Beton", "kura" = "Kura", "pedikura" = "Pedikura", "karera" = "Karera"))

ground_labels_en = c("beton" = "concrete", "kura" = "Kura", "pedikura" = "Pedikura", "karera" = "Karera") #c("Beton", "Kura", "Pedikura", "Karera")
ground_labeler_en = as_labeller(ground_labels_en)

animal_breaks = factor(animal_breaks, levels = unique(animal_breaks))
#animal_labels = factor(animal_labels, levels = unique(animal_labels))
ground_breaks = factor(ground_breaks, levels = unique(ground_breaks))
#ground_labels = factor(ground_labels, levels = unique(ground_labels))

joint_side_breaks = c("links", "rechts")


collapse_sequence = "              "
labels4 = c("R1_fitted_inv"= paste(c("\u2190 Extension    ", "Z", "      Flexion \u2192"), collapse=collapse_sequence), 
            "R2_fitted"=     paste(c("\u2190 Adduktion    ", "X", "    Abduktion \u2192"), collapse=collapse_sequence), 
            "RF_fitted"=     paste(c("\u2190 Innenrotation", "Y", "Auﬂenrotation \u2192"), collapse=collapse_sequence))
labels4_raw <- labels4
names(labels4_raw) <- c("R1_inv", "R2", "RF")
#labels4_raw = c("R1_inv"="< Extension \t Z \t Flexion >", "R2"="X (Add, Abd)", "RF"= "Y (Pro, Sup)")
labels5 = c("R1_inv"= "Z", 
            "R2"= "X", 
            "RF"= "Y")

labels6 = c("R1_inv_fitted" = "Z_fitted",
            "R2_fitted" = "X_fitted",
            "RF_fitted" = "Y_fitted")


labels_joint_grps = c("DIP" = "DIPJ", "PIP" = "PIPJ")


variabel_labels_en = c("time" = "time (%)", "rotation" = "rotation (∞)", "ground" = "flooring", "side" = "side")
collapse_sequence_en = "      "
labels4_en = c("R1_fitted_inv"= paste(c("\u2190 extension    ","Z","      flexion \u2192"), collapse=collapse_sequence_en), 
            "R2_fitted"=        paste(c("\u2190 adduction    ","X","    abduction \u2192"), collapse=collapse_sequence_en), 
            "RF_fitted"=        paste(c("\u2190 int. rotation","Y","ext. rotation \u2192"), collapse=collapse_sequence_en))

labels_joint_name_en = c("DIP links" = "DIPJ medial", "DIP rechts" = "DIPJ lateral", "PIP links" = "PIPJ medial", "PIP rechts" = "PIPJ lateral")
labels_joint_side_en = c("links" = "medial", "rechts"="lateral")


# --------------- PLOT GROUPS --------------
# These are the groups to form for plotting (will be shown together in a plot for comparison)
# ground grps to plot
ground_grps = combn(ground_breaks, 2, simplify=FALSE)
names(ground_grps) = paste(combn(ground_breaks, 2)[1,], combn(ground_breaks, 2)[2,], sep = "_vs_")
ground_grps = c(ground_grps, list(all = ground_breaks,
                                  gummi = c("kura", "pedikura", "karera"),
                                  beton = c("beton")))

ground_grps = lapply(ground_grps, factor)


#joint grps to plot
joint_grps = list(DIPs = c("DIP links", "DIP rechts"), PIPs = c("PIP links", "PIP rechts"))

#time_interval to plot
time_plot_min = -0.02
time_plot_max = 1.02

modulo_points = 60


# --------------- COLORS -----------------------------------------
qualitative_color_palette_cols = c("#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#ff7f00", "#b2df8a", "#1f78b4", "#728700")

qualitative_color_palette_ground_types <- qualitative_color_palette_cols[1:length(ground_breaks)]
names(qualitative_color_palette_ground_types) <- ground_breaks

qualitative_color_palette_animal_types <- qualitative_color_palette_cols[1:length(animal_breaks)]
names(qualitative_color_palette_animal_types) <- animal_breaks

qualitative_color_palette_sides <- qualitative_color_palette_cols[1:length(joint_side_breaks)]
names(qualitative_color_palette_sides) <- joint_side_breaks

qualitative_color_palette <- c(qualitative_color_palette_ground_types, qualitative_color_palette_animal_types)

qualitative_color_palette_animal_types_grey <- c("grey50", "grey20")
names(qualitative_color_palette_animal_types_grey) <- animal_breaks

qualitative_color_palette_ground_types_grey <- c("grey50", "grey35", "grey20", "grey5")
names(qualitative_color_palette_ground_types_grey) <- ground_breaks

qualitative_color_palette_sides_grey <- c("grey15", "grey5")
names(qualitative_color_palette_sides_grey) <- joint_side_breaks

qualitative_color_palette_grey <- c(qualitative_color_palette_ground_types_grey, 
                                    qualitative_color_palette_animal_types_grey,
                                    qualitative_color_palette_sides_grey)


qualitative_linetype_palette_ground_types <- c("solid", "C3", "dotted", "5323")
names(qualitative_linetype_palette_ground_types) <- ground_breaks

qualitative_linetype_palette_sides <- c("longdash", "dotted")
names(qualitative_linetype_palette_sides) <- joint_side_breaks

qualitative_linetype_palette <- c(qualitative_linetype_palette_ground_types, qualitative_linetype_palette_sides)


qualitative_shape_palette_ground_types <- c(16, 15, 23, 4)
names(qualitative_shape_palette_ground_types) <- ground_breaks

qualitative_shape_palette_sides <- c(16, 15)
names(qualitative_shape_palette_sides) <- joint_side_breaks

qualitative_shape_palette <- c(qualitative_shape_palette_ground_types, qualitative_shape_palette_sides)


guide_leg_6er_floor = guide_legend(
  direction = "horizontal",
  title.position = "top",
  keywidth = unit(15, units="mm")
)

guide_leg_6er_side = guide_legend(
  direction = "horizontal",
  title.position = "top",
  keywidth = unit(20, units="mm")
)

standard_width = 8
standard_height = 10
standard_units = "in"
standard_dpi = 300

plots_grey_6er_width = 140
plots_grey_6er_height = 220
plots_grey_6er_units = "mm"
plots_grey_6er_dpi = 600

# --------------- CORRECTION FACTORS ------------------------------
initial_contact_all = c("alma|beton"=95, 
                        "alma|kura"=138, 
                        "berta|beton"=72, 
                        "berta|kura"=125, 
                        "alma|pedikura"=90, 
                        "berta|pedikura"=99,
                        "alma|karera"=142,
                        "berta|karera"=177)
last_contact_all = c("alma|beton"=715, 
                     "alma|kura"=656, 
                     "berta|beton"=810, 
                     "berta|kura"=729, 
                     "alma|pedikura"=698, #previous 737 
                     "berta|pedikura"=582,
                     "alma|karera"=708,
                     "berta|karera"=733)
midstance_all = c("alma|beton"=387, 
                  "alma|kura"=363, 
                  "berta|beton"=378, 
                  "berta|kura"=383, 
                  "alma|pedikura"=350, 
                  "berta|pedikura"=292,
                  "alma|karera"=402,
                  "berta|karera"=425)


error_table_raw = data.table(RF=c(1.39,1.33,2.63,0.25,1.0,1.25), R1=c(0.64,0.83,1.45,0.33,0.56,0.53), R2 = c(0.86,0.68,4.39,0.29,1.66,1.31))
fun = max
# error = c("DIP links"=c("1"=fun(unlist(c(error_table_raw[5,2], error_table_raw[3,2]))), 
#                         "2"=fun(unlist(c(error_table_raw[5,3], error_table_raw[3,3]))), 
#                         "F"=fun(unlist(c(error_table_raw[5,1], error_table_raw[3,1])))), 
#           "PIP links"=c("1"=fun(unlist(c(error_table_raw[1,2], error_table_raw[3,2]))), 
#                         "2"=fun(unlist(c(error_table_raw[1,3], error_table_raw[3,3]))), 
#                         "F"=fun(unlist(c(error_table_raw[1,1], error_table_raw[3,1])))), 
#           "DIP rechts"=c("1"=fun(unlist(c(error_table_raw[6,2], error_table_raw[4,2]))), 
#                          "2"=fun(unlist(c(error_table_raw[6,3], error_table_raw[4,3]))), 
#                          "F"=fun(unlist(c(error_table_raw[6,1], error_table_raw[4,1])))), 
#           "PIP rechts"=c("1"=fun(unlist(c(error_table_raw[2,2], error_table_raw[4,2]))), 
#                          "2"=fun(unlist(c(error_table_raw[2,3], error_table_raw[4,3]))), 
#                          "F"=fun(unlist(c(error_table_raw[2,1], error_table_raw[4,1])))))

error = c("DIP links"=c("1"=2.63, 
                        "2"=1.45, 
                        "F"=4.39), 
          "PIP links"=c("1"=2.63, 
                        "2"=1.45, 
                        "F"=4.39), 
          "DIP rechts"=c("1"=2.38, 
                         "2"=1.46, 
                         "F"=3.92), 
          "PIP rechts"=c("1"=2.38, 
                         "2"=1.46, 
                         "F"=3.92))


# --------------- DATA AGGREGATION -------------------------------------
data = data.table()
data_per_frame_wide = data.table()
RF_correction = data.table()
for(d_first in list.dirs(recursive = FALSE))
{
  split = strsplit(d_first, "/")[[1]]
  animal_name = split[length(split)]
  
  for(f_cs in list.files(path=d_first, pattern = "cs-.*[.]csv$", full.names = TRUE))
  {
     csv_JCS =  data.table(read.csv(f_cs))
     RF_correction_value = unlist(csv_JCS[variable=="RF","value"][1])
     RF_correction_value =as.numeric(levels(RF_correction_value))[RF_correction_value]
     RF_correction_axis1 = unlist(csv_JCS[variable=="objs[0]","value"][1])
     RF_correction_axis1 =as.character(levels(RF_correction_axis1))[RF_correction_axis1]
     RF_correction_axis2 = unlist(csv_JCS[variable=="objs[1]","value"][1])
     RF_correction_axis2 =as.character(levels(RF_correction_axis2))[RF_correction_axis2]
     RF_correction=rbind(RF_correction, data.table(animal_name=animal_name, 
                                                   axis1 = strsplit(RF_correction_axis1, ":")[[1]][1],
                                                   axis2 = strsplit(RF_correction_axis2, ":")[[1]][1],
                                                   value = RF_correction_value))
  }
  
  for(d_second in list.dirs(path = d_first, recursive=FALSE))
  {
    split = strsplit(d_second, "/")[[1]]
    ground_type = split[length(split)]
    
    initial_contact = initial_contact_all[paste(tolower(animal_name), tolower(ground_type), sep="|")]
    last_contact = last_contact_all[paste(tolower(animal_name), tolower(ground_type), sep="|")]
    
    for(f in list.files(path = d_second, pattern = "rot-.*[.]csv$", full.names = TRUE))
    {
      split = strsplit(f, "/")[[1]]
      split = strsplit(split[length(split)], "-")[[1]]
      axis2 = split[length(split)]
      axis1 = split[length(split)-1]
      split = gregexpr(".", axis2, fixed=TRUE)[[1]]
      axis2 = substr(axis2, 0, split[length(split)]-1)
      x = data.table(read.csv(f))
      diff = data.table()
      for(colNo in 5:7)
      {
        x[,paste("diff", substr(colnames(x)[colNo], 0, 1), sep=""):= x[, colNo, with = FALSE]-x[, colNo+3, with = FALSE]]
      }
      # x[,"distance":=sqrt(diffX^2+diffY^2+diffZ^2)]
      
      x[,"corrected_frame":=(frame-initial_contact)]
      
      x[,"time":=(frame-initial_contact)/(last_contact-initial_contact)]
      
      x[,"RF":=RF-unlist(RF_correction[animal_name==.GlobalEnv$animal_name&axis1==.GlobalEnv$axis1&axis2==.GlobalEnv$axis2, "value"])]
      
      loess_vars = list()
      for(var in fitting_vars)
      {
        loess_vars[[var]] <- loess(paste(var, "~time", sep=""), data=x, degree=2, span = span_for_fit)
        x[, paste(var, "fitted", sep="_"):=loess_vars[[var]]$fitted]
        x[, paste(var, "grad", sep="_"):= c(diff(loess_vars[[var]]$fitted)/diff(x[["time"]]), NA)] #sfsmisc::D1D2(loess_vars[[var]]$x, loess_vars[[var]]$fitted, deriv=c(1))$D1] 
      }

      y = melt(x, id.vars = c("time"), measure.vars = data_vars)
      y[, "animal_name":=factor(animal_name)]
      y[, "ground_type":=factor(ground_type)]
      y[, "axis1":=factor(axis1)]
      y[, "axis2":=factor(axis2)]
      data = rbind(data, y)
      
      
      x[, "animal_name":=factor(animal_name)]
      x[, "ground_type":=factor(ground_type)]
      x[, "axis1":=factor(axis1)]
      x[, "axis2":=factor(axis2)]
      data_per_frame_wide = rbind(data_per_frame_wide, x)
    }
  }
} 

#ggplot(data, aes(x=frame, y=value, color = ground_type, group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_line()+geom_smooth(span = 0.4, color = "grey")+facet_grid(variable~axis1)
#ggplot(data[variable %in% plot_vars, ], aes(x=time, y=value, color = ground_type, group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_vline(xintercept = c(0,1), color="grey")+geom_line()+facet_grid(variable~axis1, scales = "free_y")


# ggplot(data[variable %in% plot_vars, ], aes(x=time, y=value, color = interaction(animal_name, ground_type), group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_vline(xintercept = c(0,1), color="grey")+geom_line()+facet_grid(variable~axis1, scales = "free_y")+scale_color_brewer(palette = "Paired")
# ggplot(data[variable %in% plot_vars_2, ], aes(x=time, y=value, color = interaction(animal_name, ground_type), group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_vline(xintercept = c(0,1), color="grey")+geom_line()+facet_grid(variable~axis1, scales = "free_y")+scale_color_brewer(palette = "Paired")
# ggplot(data[variable %in% plot_vars_2, ], aes(x=time, y=value, color = ground_type, group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_vline(xintercept = c(0,1), color="grey")+geom_line()+facet_grid(list("variable", c("animal_name", "axis1")), scales = "free_y")+scale_color_brewer(palette = "Set1")
# ggplot(data[variable %in% plot_vars, ], aes(x=time, y=value, color = ground_type, group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_vline(xintercept = c(0,1), color="grey")+geom_line()+facet_grid(list("variable", c("animal_name", "axis1")), scales = "free_y")+scale_color_brewer(palette = "Set1")
# ggplot(data[variable %in% plot_vars_2, ], aes(x=time, y=value, color = ground_type, group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_vline(xintercept = c(0,1), color="grey")+geom_line()+facet_grid(list("variable", c("animal_name", "axis1")), scales = "free_y")+scale_color_brewer(palette = "Set1")
# ggplot(data[variable %in% plot_vars_3, ], aes(x=time, y=value, color = ground_type, group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_vline(xintercept = c(0,1), color="grey")+geom_line()+facet_grid(list("variable", c("animal_name", "axis1")), scales = "free_y")+scale_color_brewer(palette = "Set1")
# ggplot(data[variable %in% plot_vars_3 & ground_type=="beton", ], aes(x=time, y=value, color = animal_name, group = interaction(animal_name, ground_type, variable, axis1)))+geom_hline(yintercept = 0, color = "grey")+geom_vline(xintercept = c(0,1), color="grey")+geom_line()+facet_grid(list("variable", c("axis1")), scales = "free_y")+scale_color_brewer(palette = "Set1")



data[(axis1 == "Klauenbein_links" & axis2 == "Kronbein_links") | (axis2 == "Klauenbein_links" & axis1 == "Kronbein_links") , joint_name:="DIP links"]
data[(axis1 == "Kronbein_links" & axis2 == "Fesselbein_links") | (axis2 == "Kronbein_links" & axis1 == "Fesselbein_links") , joint_name:="PIP links"]
data[(axis1 == "Klauenbein_rechts" & axis2 == "Kronbein_rechts") | (axis2 == "Klauenbein_rechts" & axis1 == "Kronbein_rechts") , joint_name:="DIP rechts"]
data[(axis1 == "Kronbein_rechts" & axis2 == "Fesselbein_rechts") | (axis2 == "Kronbein_rechts" & axis1 == "Fesselbein_rechts") , joint_name:="PIP rechts"]

data[,joint_group := tstrsplit(joint_name, " ", fixed = T)[1]]
data[,joint_side := tstrsplit(joint_name, " ", fixed = T)[2]]


copy <- data[variable == "R1"]
copy[, value :=-value]
copy[, variable :="R1_inv"]
data= rbind(copy, data)

copy <- data[variable == "R1_fitted"]
copy[, value :=-value]
copy[, variable :="R1_fitted_inv"]
data= rbind(copy, data)


data_per_frame_wide[(axis1 == "Klauenbein_links" & axis2 == "Kronbein_links") | (axis2 == "Klauenbein_links" & axis1 == "Kronbein_links") , joint_name:="DIP links"]
data_per_frame_wide[(axis1 == "Kronbein_links" & axis2 == "Fesselbein_links") | (axis2 == "Kronbein_links" & axis1 == "Fesselbein_links") , joint_name:="PIP links"]
data_per_frame_wide[(axis1 == "Klauenbein_rechts" & axis2 == "Kronbein_rechts") | (axis2 == "Klauenbein_rechts" & axis1 == "Kronbein_rechts") , joint_name:="DIP rechts"]
data_per_frame_wide[(axis1 == "Kronbein_rechts" & axis2 == "Fesselbein_rechts") | (axis2 == "Kronbein_rechts" & axis1 == "Fesselbein_rechts") , joint_name:="PIP rechts"]

data_per_frame_wide[,joint_group := tstrsplit(joint_name, " ", fixed = T)[1]]
data_per_frame_wide[,joint_side := tstrsplit(joint_name, " ", fixed = T)[2]]

data_per_frame_wide[, R1_inv := -R1]
data_per_frame_wide[, R1_fitted_inv := -R1_fitted]


ax_names_to_index = c("R1_fitted_inv"="1", "R2_fitted"="2", "RF_fitted"= "F")
joint_names = c("PIP links","PIP rechts", "DIP links", "DIP rechts")
for(joint in joint_names)
{
  for(ax in names(ax_names_to_index))
  {
    data[variable == ax & joint_name== joint, value_min_error := value - error[paste(joint, ax_names_to_index[ax], sep=".")]]
    data[variable == ax & joint_name== joint, value_max_error := value + error[paste(joint, ax_names_to_index[ax], sep=".")]]
  }
}

data_replaced <- data[(variable %in% plot_vars_4_raw), ]
data_replaced[variable == plot_vars_4_raw[1], variable := plot_vars_4[1]]
data_replaced[variable == plot_vars_4_raw[2], variable := plot_vars_4[2]]
data_replaced[variable == plot_vars_4_raw[3], variable := plot_vars_4[3]]
data_replaced[, pt_seq := seq(time), by = c("variable", "animal_name", "ground_type", "joint_name")]



# --------------- INTERPOLATION AND CALCULATION OF MEAN -------------------------------------

xout = sort(unique(unlist(data[, "time"])))
xout = xout[xout>-0.001&xout<1.001]
interpolations = data.table() #time=c(), ground_type=factor(), joint_name=factor(), variable = factor(), value_lin = c(), value_spline=c())
for(ground in levels(data$ground_type))
{
  for(animal in levels(data$animal_name))
  {
    for(joint in levels(factor(data$joint_name)))
    {
      for(var in plot_vars_4_raw)#levels(data$variable))
      {
        d_for_fit = data[(animal_name==animal)&(variable==var)&(ground_type==ground)&(joint_name==joint), ]
        lin_int = approx(x=unlist(d_for_fit[, "time"]), y=unlist(d_for_fit[, "value"]), xout=xout)
        # lin_int_fit = loess("val~time", data=data.frame(time=xout, val=lin_int$y), degree=2, span = span_for_fit)
        spl_int = spline(x=unlist(d_for_fit[, "time"]), y=unlist(d_for_fit[, "value"]), xout=xout)
        # spl_int_fit = loess("val~time", data=data.frame(time=xout, val=spl_int$y), degree=2, span = span_for_fit)
        interpolations=rbind(interpolations, data.table(time=xout, variable=var, 
                                                        lin_int = lin_int$y, spl_int = spl_int$y, 
                                                        # lin_int_fit = lin_int_fit$y, spl_int_fit = spl_int_fit$y,
                                                        animal_name=animal, 
                                                        ground_type=ground, 
                                                        joint_name=joint))
      }
    }
  }
}
interpolations_means = interpolations[, list(lin_int_mean=mean(lin_int), spl_int_mean= mean(spl_int)), by=list(time,variable,ground_type, joint_name) ]

for(ground in levels(as.factor(interpolations_means$ground_type)))
{
  for(joint in levels(factor(data$joint_name)))
  {
    for(var in levels(as.factor(interpolations_means$variable)))#levels(data$variable))
    {
      d_for_fit = interpolations_means[(variable==var)&(ground_type==ground)&(joint_name==joint), ]
      lin_int_loess = loess("lin_int_mean~time", data=d_for_fit, degree=2, span = span_for_fit)
      spl_int_loess = loess("spl_int_mean~time", data=d_for_fit, degree=2, span = span_for_fit)
      interpolations_means[(variable==var)&(ground_type==ground)&(joint_name==joint), 
                           lin_int_mean_fit:=lin_int_loess$fitted]
      interpolations_means[(variable==var)&(ground_type==ground)&(joint_name==joint), 
                           spl_int_mean_fit := spl_int_loess$fitted]
    }
  }
}

interpolations_means[, value:=spl_int_mean_fit]
ax_names_to_index = c("R1_inv"="1", "R2"="2", "RF"= "F")
for(joint in joint_names)
{
  for(ax in names(ax_names_to_index))
  {
    interpolations_means[variable == ax & joint_name== joint, value_min_error := value - error[paste(joint, ax_names_to_index[ax], sep=".")]]
    interpolations_means[variable == ax & joint_name== joint, value_max_error := value + error[paste(joint, ax_names_to_index[ax], sep=".")]]
  }
}
interpolations_means[, variable:=factor(interpolations_means$variable, levels = c("R2", "RF", "R1_inv"))]


# --------------- CALCULATION OF EXTREMA AND MIDSTANCE -------------------------

# mins = copy(data[time<=1&time>=0, .SD[which.min(value)], by = list(variable, animal_name, ground_type, joint_name)])
# maxs = copy(data[time<=1&time>=0, .SD[which.max(value)], by = list(variable, animal_name, ground_type, joint_name)])
# extrema = rbind(cbind(mins, type = "min"),cbind(maxs, type = "max"))
# extrema[variable%in%names(labels5), variable:=labels5[as.character(unlist(variable))]]
# #extrema[variable%in%names(labels5),]$variable=labels5[as.character(unlist(extrema[variable%in%names(labels5),]$variable))]
# extrema_wide = dcast(extrema, animal_name+ ground_type+ joint_name~variable+type, value.var = c("value", "time"))
# extrema_out = extrema_wide[, c("animal_name", "ground_type", "joint_name", 
#                                   apply(expand.grid(c("value", "time"),
#                                     apply(expand.grid(labels5, c("min", "max")), 1, paste, collapse="_")), 1, paste, collapse="_")), 
#                            with=FALSE]
# write.csv(extrema_out, "extrema.csv")
# write.csv2(extrema_out, "extrema_for_excel.csv")
# 
# for(ax in labels5)
# {
#   diff_rom = extrema_out[, paste(c("value",ax,"max"), collapse="_"), with=FALSE] - extrema_out[, paste(c("value",ax,"min"), collapse="_"), with=FALSE]
#   extrema_out[,paste(c(ax, "_rom"), collapse=""):=diff_rom]
# }

fun_extrema <- function(data)
{
  mins = copy(data[time<=1&time>=0, .SD[which.min(value)], by = list(variable, animal_name, ground_type, joint_name)])
  maxs = copy(data[time<=1&time>=0, .SD[which.max(value)], by = list(variable, animal_name, ground_type, joint_name)])
  extrema = rbind(cbind(mins, type = "min"),cbind(maxs, type = "max"))
  extrema[variable%in%names(labels5), variable:=labels5[as.character(unlist(variable))]]
  #extrema[variable%in%names(labels5),]$variable=labels5[as.character(unlist(extrema[variable%in%names(labels5),]$variable))]
  extrema_wide = dcast(extrema, animal_name+ ground_type+ joint_name~variable+type, value.var = c("value", "time"))
  extrema_out = extrema_wide[, c("animal_name", "ground_type", "joint_name", 
                                 apply(expand.grid(c("value", "time"),
                                                   apply(expand.grid(labels5, c("min", "max")), 1, paste, collapse="_")), 1, paste, collapse="_")), 
                             with=FALSE]
  for(ax in labels5)
  {
    diff_rom = extrema_out[, paste(c("value",ax,"max"), collapse="_"), with=FALSE] - extrema_out[, paste(c("value",ax,"min"), collapse="_"), with=FALSE]
    extrema_out[,paste(c(ax, "_rom"), collapse=""):=diff_rom]
  }
  
  return(extrema_out)
}


fun_local_extrema_one_axis <- function(times, values, winsize)
{
  order_times = order(times)
  times = times[order_times]
  values = values[order_times]
  
  mat_ret_cpp_min = slidingWindowMin(times, values, winsize)
  mat_ret_cpp_inverted_min = slidingWindowMin(times, -values, winsize)
  mat_ret_cpp_inverted_min[,1]=-mat_ret_cpp_inverted_min[,1] 
  
  is_extr_min = (mat_ret_cpp_min[,3] == seq(nrow(mat_ret_cpp_min)))
  is_extr_max = (mat_ret_cpp_inverted_min[,3] == seq(nrow(mat_ret_cpp_inverted_min)))
  
  extr_min_idc = which(is_extr_min)
  extr_max_idc = which(is_extr_max)
  
  
  df_ret = rbind(data.table(type = "min", time=mat_ret_cpp_min[extr_min_idc, 2], value = mat_ret_cpp_min[extr_min_idc, 1]),
                 data.table(type = "max", time=mat_ret_cpp_inverted_min[extr_max_idc, 2], value = mat_ret_cpp_inverted_min[extr_max_idc, 1]))
  
  return(df_ret)
}

fun_local_extrema <- function(data, winsize)
{
  ret = FALSE
  for(animal in unique(data[,animal_name]))
  {
    for(ground in unique(data[animal_name == animal,ground_type]))
    {
      for(joint in unique(data[animal_name == animal & ground_type == ground, joint_name]))
      {
        for(axis in unique(data[animal_name == animal & ground_type == ground & joint_name == joint, variable]))
        {
          times_single_axis = data[animal_name == animal & 
                                     ground_type == ground & 
                                     joint_name == joint &
                                     variable == axis, time]
          values_single_axis = data[animal_name == animal & 
                                      ground_type == ground & 
                                      joint_name == joint &
                                      variable == axis, value]
          tbl_single_axis = fun_local_extrema_one_axis(times_single_axis, values_single_axis, winsize)
          tbl_single_axis = cbind(animal = animal, ground = ground, joint = joint, axis=axis, tbl_single_axis)
          
          if(!is.data.table(ret))
          {
            ret = tbl_single_axis
          } else {
            ret = rbind(ret, tbl_single_axis)
          }
          
        }
      }
    }
  }
  return(ret)
}


extrema_out <- fun_extrema(data=data)
write.csv(extrema_out, paste(tables_dir, "/", "extrema.csv", sep=""))
write.csv2(extrema_out, paste(tables_dir, "/", "extrema_for_excel.csv", sep=""))

# extrema_local_out = fun_extrema(data=data[ground_type=="beton"&joint_name=="DIP rechts"&time<=0.7&time>=0.5,])[,c("animal_name", "ground_type", "joint_name", "value_Z_min", "time_Z_min")]
# write.csv(extrema_local_out, paste(tables_dir, "/", "extrema_local.csv", sep=""))
# write.csv2(extrema_local_out, paste(tables_dir, "/", "extrema_local_for_excel.csv", sep=""))


extrema_local = fun_local_extrema(data = data[variable %in% c(names(labels5)) & time>=0 & time<=1, ], winsize = 0.2)
extrema_local[, axis := c(labels5, labels6)[axis]]
setorderv(extrema_local, c("animal", "ground", "joint", "axis", "type", "time"))
write.csv(extrema_local, paste(tables_dir, "/", "extrema_local.csv", sep=""))


fun_extrema_interpolations <- function(data)
{
  mins = copy(data[time<=1&time>=0, .SD[which.min(value)], by = list(variable, ground_type, joint_name)])
  maxs = copy(data[time<=1&time>=0, .SD[which.max(value)], by = list(variable, ground_type, joint_name)])
  extrema = rbind(cbind(mins, type = "min"),cbind(maxs, type = "max"))
  extrema[variable%in%names(labels5), variable:=labels5[as.character(unlist(variable))]]
  #extrema[variable%in%names(labels5),]$variable=labels5[as.character(unlist(extrema[variable%in%names(labels5),]$variable))]
  extrema_wide = dcast(extrema, ground_type+ joint_name~variable+type, value.var = c("value", "time"))
  extrema_out = extrema_wide[, c("ground_type", "joint_name", 
                                 apply(expand.grid(c("value", "time"),
                                                   apply(expand.grid(labels5, c("min", "max")), 1, paste, collapse="_")), 1, paste, collapse="_")), 
                             with=FALSE]
  for(ax in labels5)
  {
    diff_rom = extrema_out[, paste(c("value",ax,"max"), collapse="_"), with=FALSE] - extrema_out[, paste(c("value",ax,"min"), collapse="_"), with=FALSE]
    extrema_out[,paste(c(ax, "_rom"), collapse=""):=diff_rom]
  }
  
  return(extrema_out)
}

extrema_means_out <- fun_extrema_interpolations(data=interpolations_means)
write.csv(extrema_means_out, paste(tables_dir, "/", "extrema_means.csv", sep=""))
write.csv2(extrema_means_out, paste(tables_dir, "/", "extrema_means_for_excel.csv", sep=""))


midstance_tbl = data.table()
for(animal in levels(data$animal_name))
{
  for(ground in levels(data$ground_type))
  {
    vidx = paste(c(animal, ground),collapse="|")
    time_m = as.double((midstance_all[vidx]-initial_contact_all[vidx]))/(last_contact_all[vidx]-initial_contact_all[vidx])
    midstance_tbl = rbind(midstance_tbl, data.table(animal_name=animal, ground_type=ground, time_midstance =time_m))
  }
}

combs_animal_names = combn(midstance_tbl[["animal_name"]], 2)
combs_ground_types = combn(midstance_tbl[["ground_type"]], 2)
combs_midstance_times = combn(midstance_tbl[["time_midstance"]],2)

midstance_diffs = data.table(animal_name_1 = c(combs_animal_names[1,]),
                             ground_type_1 = c(combs_ground_types[1,]),
                             midstance_time_1 = c(combs_midstance_times[1,]),
                             animal_name_2 = c(combs_animal_names[2,]),
                             ground_type_2 = c(combs_ground_types[2,]),
                             midstance_time_2 = c(combs_midstance_times[2,]) #,
                             #time_midstance_diffs = dist(midstance_tbl[["time_midstance"]])
                             )


midstance_diffs[, time_midstance_diffs:= abs(midstance_time_1-midstance_time_2)]
midstance_diffs[, time_diffs_frames_1 := time_midstance_diffs * unname((last_contact_all-initial_contact_all)[paste(midstance_diffs$animal_name_1, midstance_diffs$ground_type_1, sep="|")])]
midstance_diffs[, time_diffs_frames_2 := time_midstance_diffs * unname((last_contact_all-initial_contact_all)[paste(midstance_diffs$animal_name_2, midstance_diffs$ground_type_2, sep="|")])]

write.csv(midstance_diffs, paste(tables_dir, "/", "midstance_diffs.csv", sep=""))
write.table(midstance_diffs, paste(tables_dir, "/", "midstance_diffs_for_excel.csv", sep=""), dec=",", sep=";", row.names = FALSE)


# --------------- OUTPUT TABLES ----------------------------
output_coloumns = c("animal_name" = "animal_name", 
                    "ground_type" = "ground_type", 
                    "joint_name" = "joint_name", 
                    "frame" = "frame", 
                    "corrected_frame" = "corrected_frame",
                    "time" = "time",
                    "RAbAd" = "R2",
                    "RProSup" = "RF",
                    "RFlexEx"= "R1_inv",
                    "RAbAd_fitted" = "R2_fitted",
                    "RProSup_fitted" = "RF",
                    "RFlexEx_fitted"= "R1_fitted_inv")

output_coloumns_inverse = names(output_coloumns)
names(output_coloumns_inverse) = output_coloumns

data_per_frame_wide_file_output = data_per_frame_wide[, output_coloumns, with=FALSE]
colnames(data_per_frame_wide_file_output) = names(output_coloumns)



data_means_per_frame_file_output = melt(data_per_frame_wide_file_output, 
                                        measure.vars=c("RAbAd",
                                                       "RProSup",
                                                       "RFlexEx",
                                                       "RAbAd_fitted",
                                                       "RProSup_fitted",
                                                       "RFlexEx_fitted"))

data_means_per_raw_frame_file_output = data_means_per_frame_file_output[, list(value=mean(value)), by = c("joint_name", "ground_type", "frame", "variable")]
data_means_per_corrected_frame_file_output = data_means_per_frame_file_output[, list(value=mean(value)), by = c("joint_name", "ground_type", "corrected_frame", "variable")]

data_means_per_raw_frame_file_output = dcast(data_means_per_raw_frame_file_output, joint_name + ground_type + frame ~ variable, value.var = "value")
data_means_per_corrected_frame_file_output = dcast(data_means_per_corrected_frame_file_output, joint_name + ground_type + corrected_frame ~ variable, value.var = "value")

data_means_interpolation_file_output = copy(interpolations_means)
data_means_interpolation_file_output[,variable := output_coloumns_inverse[as.character(variable)]]
data_means_interpolation_file_output = dcast(melt(data_means_interpolation_file_output, 
                                                  id.vars = c("time", "ground_type", "joint_name", "variable")), 
                                             time+ground_type+joint_name~variable+variable.1)


write.csv(data_per_frame_wide_file_output, paste(tables_dir, "/", "data.csv", sep=""))
# write.csv2(data_per_frame_wide_file_output, paste(tables_dir, "/", "data_for_excel.csv", sep=""))

write.csv(data_means_per_corrected_frame_file_output, paste(tables_dir, "/", "data_means_per_frame.csv", sep = ""))
# write.csv2(data_means_per_corrected_frame_file_output, paste(tables_dir, "/", "data_means_per_frame_for_excel.csv", sep = ""))

write.csv(data_means_interpolation_file_output, paste(tables_dir, "/", "data_means_interpolation.csv", sep=""))
# write.csv2(data_means_interpolation_file_output, paste(tables_dir, "/", "data_means_interpolation_for_excel.csv", sep=""))



# --------------- PLOTS -------------------------------------
for(animal in levels(data$animal_name))
{
  for(joint_grp_name in names(joint_grps))
  {
    joint_grp = joint_grps[[joint_grp_name]]
    g <- ggplot(data[(variable %in% plot_vars_4) & animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks &
                       time >= time_plot_min&
                       time <= time_plot_max, ],
           aes(x=time*100, y=value, color = ground_type, fill=ground_type, group = interaction(animal_name, ground_type, variable, joint_name)))+
      geom_vline(data=midstance_tbl[animal_name==animal & ground_type%in%ground_breaks,], aes(xintercept=time_midstance*100, color=ground_type), linetype=6, show.legend=FALSE)+
      geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
      geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
      geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
      geom_line(data=data_replaced[animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks, ])+
      facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
      scale_color_manual(name="Bodentyp", breaks = ground_breaks, labels=ground_labels, values = qualitative_color_palette)+
      scale_fill_manual(name = "Bodentyp", breaks = ground_breaks, labels=ground_labels, values = qualitative_color_palette)+
      scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = "Rotation (∞)")+
      theme_bw()+
      theme(legend.justification = c(1,0))
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", joint_grp_name , ".pdf"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", joint_grp_name , ".png"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           dpi = standard_dpi)
  }
}

for(animal in levels(data$animal_name))
{
  for(joint_grp_name in names(joint_grps))
  {
    joint_grp = joint_grps[[joint_grp_name]]
    

    
    g <- ggplot(data[(variable %in% plot_vars_4) & animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks&
                       time >= time_plot_min&
                       time <= time_plot_max, ],
                aes(x=time*100, y=value, 
                    color = ground_type, 
                    linetype=ground_type, 
                    shape=ground_type,
                    #fill=ground_type, 
                    group = interaction(animal_name, ground_type, variable, joint_name)))+
      #geom_vline(data=midstance_tbl[animal_name==animal & ground_type%in%ground_breaks,], aes(xintercept=time_midstance*100, color=ground_type), linetype=6, show.legend=FALSE)+
      #geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
      #geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
      #geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
      geom_line(data=data_replaced[animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks&
                                     time >= time_plot_min&
                                     time <= time_plot_max, ])+
      #geom_point(data=data_replaced[animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks & (pt_seq %% modulo_points == 0)&
      #                               time >= time_plot_min&
      #                                time <= time_plot_max, ],
      #           size = 0.7)+
      facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4_en, joint_name = labels_joint_name_en))+
      scale_color_manual(name=variabel_labels_en["ground"], breaks = ground_breaks, labels=ground_labels_en, values = qualitative_color_palette_grey)+
      scale_fill_manual(name = variabel_labels_en["ground"], breaks = ground_breaks, labels=ground_labels_en, values = qualitative_color_palette_grey)+
      scale_linetype_manual(name=variabel_labels_en["ground"], breaks = ground_breaks, labels=ground_labels_en, values = qualitative_linetype_palette)+
      scale_shape_manual(name=variabel_labels_en["ground"], breaks = ground_breaks, labels=ground_labels_en, values = qualitative_shape_palette)+
      scale_x_continuous(name = variabel_labels_en["time"], limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = variabel_labels_en["rotation"])+
      theme_bw()+
      theme(legend.position="bottom")+
      guides(linetype = guide_leg_6er_floor , color = guide_leg_6er_floor , fill = guide_leg_6er_floor , shape = guide_leg_6er_floor )
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", joint_grp_name , "_grey_en_without_errors_without_midstances.pdf"), collapse=""), 
           width=plots_grey_6er_width, 
           height=plots_grey_6er_height, 
           units = plots_grey_6er_units, 
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", joint_grp_name , "_grey_en_without_errors_without_midstances.png"), collapse=""), 
           width=plots_grey_6er_width, 
           height=plots_grey_6er_height, 
           units = plots_grey_6er_units,
           dpi = plots_grey_6er_dpi)
    
    
    
    g <- ggplot(data[(variable %in% plot_vars_4) & animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks&
                       time >= time_plot_min&
                       time <= time_plot_max, ],
                aes(x=time*100, y=value, 
                    color = ground_type, 
                    linetype=ground_type, 
                    shape=ground_type,
                    #fill=ground_type, 
                    group = interaction(animal_name, ground_type, variable, joint_name)))+
      geom_vline(data=midstance_tbl[animal_name==animal & ground_type%in%ground_breaks,], 
                 aes(xintercept=time_midstance*100, linetype = ground_type),
                 color = "grey70",
                 show.legend=FALSE)+
      #geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
      #geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
      #geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
      geom_line(data=data_replaced[animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks&
                                     time >= time_plot_min&
                                     time <= time_plot_max, ])+
      #geom_point(data=data_replaced[animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks & (pt_seq %% modulo_points == 0)&
      #                               time >= time_plot_min&
      #                                time <= time_plot_max, ],
      #           size = 0.7)+
      facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4_en, joint_name = labels_joint_name_en))+
      scale_color_manual(name=variabel_labels_en["ground"], breaks = ground_breaks, labels=ground_labels_en, values = qualitative_color_palette_grey)+
      scale_fill_manual(name = variabel_labels_en["ground"], breaks = ground_breaks, labels=ground_labels_en, values = qualitative_color_palette_grey)+
      scale_linetype_manual(name=variabel_labels_en["ground"], breaks = ground_breaks, labels=ground_labels_en, values = qualitative_linetype_palette)+
      scale_shape_manual(name=variabel_labels_en["ground"], breaks = ground_breaks, labels=ground_labels_en, values = qualitative_shape_palette)+
      scale_x_continuous(name = variabel_labels_en["time"], limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = variabel_labels_en["rotation"])+
      theme_bw()+
      theme(legend.position="bottom")+
      guides(linetype = guide_leg_6er_floor , color = guide_leg_6er_floor , fill = guide_leg_6er_floor , shape = guide_leg_6er_floor )
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", joint_grp_name , "_grey_en_without_errors_with_midstances.pdf"), collapse=""), 
           width=plots_grey_6er_width, 
           height=plots_grey_6er_height, 
           units = plots_grey_6er_units,
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", joint_grp_name , "_grey_en_without_errors_with_midstances.png"), collapse=""), 
           width=plots_grey_6er_width, 
           height=plots_grey_6er_height, 
           units = plots_grey_6er_units,
           dpi = plots_grey_6er_dpi)
  }
}


for(animal in levels(data$animal_name))
{
  for(ground in levels(data$ground_type))
  {
    joint_grp = joint_grps[[joint_grp_name]]
    g <- ggplot(data[(variable %in% plot_vars_4) & animal_name==animal & ground_type==ground&
                       time >= time_plot_min&
                       time <= time_plot_max, ],
                aes(x=time*100, y=value, 
                    color = joint_side, 
                    linetype=joint_side, 
                    shape=joint_side,
                    fill=joint_side, 
                    group = interaction(animal_name, ground_type, variable, joint_name)))+
      #geom_vline(data=midstance_tbl[animal_name==animal & ground_type%in%ground_breaks,], aes(xintercept=time_midstance*100, color=ground_type), linetype=6, show.legend=FALSE)+
      #geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
      #geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
      #geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
      geom_line(data=data_replaced[animal_name==animal & ground_type==ground &
                                     time >= time_plot_min&
                                     time <= time_plot_max, ])+
      #geom_point(data=data_replaced[animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks & (pt_seq %% modulo_points == 0)&
      #                               time >= time_plot_min&
      #                                time <= time_plot_max, ],
      #           size = 0.7)+
      facet_grid(list("variable", c("joint_group")), scales = "free_y", labeller=labeller(variable = labels4_en, joint_group = labels_joint_grps))+
      scale_color_manual(name=variabel_labels_en["side"], breaks = joint_side_breaks, labels=labels_joint_side_en, values = qualitative_color_palette_grey)+
      scale_fill_manual(name = variabel_labels_en["side"], breaks = joint_side_breaks, labels=labels_joint_side_en, values = qualitative_color_palette_grey)+
      scale_linetype_manual(name=variabel_labels_en["side"], breaks = joint_side_breaks, labels=labels_joint_side_en, values = qualitative_linetype_palette)+
      scale_shape_manual(name=variabel_labels_en["side"], breaks = joint_side_breaks, labels=labels_joint_side_en, values = qualitative_shape_palette)+
      scale_x_continuous(name = variabel_labels_en["time"], limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = variabel_labels_en["rotation"])+
      theme_bw()+
      theme(legend.position="bottom")+
      guides(linetype = guide_leg_6er_side , color = guide_leg_6er_side , fill = guide_leg_6er_side , shape = guide_leg_6er_side)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", ground , "_grey_en_without_errors_without_midstances.pdf"), collapse=""), 
           width=plots_grey_6er_width, 
           height=plots_grey_6er_height, 
           units = plots_grey_6er_units,
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", ground , "_grey_en_without_errors_without_midstances.png"), collapse=""), 
           width=plots_grey_6er_width, 
           height=plots_grey_6er_height, 
           units = plots_grey_6er_units,
           dpi = plots_grey_6er_dpi)
    
    
    g <- ggplot(data[(variable %in% plot_vars_4) & animal_name==animal & ground_type==ground&
                       time >= time_plot_min&
                       time <= time_plot_max, ],
                aes(x=time*100, y=value, 
                    color = joint_side, 
                    linetype=joint_side, 
                    shape=joint_side,
                    fill=joint_side, 
                    group = interaction(animal_name, ground_type, variable, joint_name)))+
      geom_vline(data=midstance_tbl[animal_name==animal & ground_type==ground,], 
                 aes(xintercept=time_midstance*100, color=ground_type),
                 linetype = 6,
                 color = "grey70",
                 show.legend=FALSE)+
      #geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
      #geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
      #geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
      geom_line(data=data_replaced[animal_name==animal & ground_type==ground &
                                     time >= time_plot_min&
                                     time <= time_plot_max, ])+
      #geom_point(data=data_replaced[animal_name==animal & joint_name%in%joint_grp & ground_type%in%ground_breaks & (pt_seq %% modulo_points == 0)&
      #                               time >= time_plot_min&
      #                                time <= time_plot_max, ],
      #           size = 0.7)+
      facet_grid(list("variable", c("joint_group")), scales = "free_y", labeller=labeller(variable = labels4_en, joint_group = labels_joint_grps))+
      scale_color_manual(name=variabel_labels_en["side"], breaks = joint_side_breaks, labels=labels_joint_side_en, values = qualitative_color_palette_grey)+
      scale_fill_manual(name = variabel_labels_en["side"], breaks = joint_side_breaks, labels=labels_joint_side_en, values = qualitative_color_palette_grey)+
      scale_linetype_manual(name=variabel_labels_en["side"], breaks = joint_side_breaks, labels=labels_joint_side_en, values = qualitative_linetype_palette)+
      scale_shape_manual(name=variabel_labels_en["side"], breaks = joint_side_breaks, labels=labels_joint_side_en, values = qualitative_shape_palette)+
      scale_x_continuous(name = variabel_labels_en["time"], limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = variabel_labels_en["rotation"])+
      theme_bw()+
      theme(legend.position="bottom")+
      guides(linetype = guide_leg_6er_side , color = guide_leg_6er_side , fill = guide_leg_6er_side , shape = guide_leg_6er_side)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", ground , "_grey_en_without_errors_with_midstance.pdf"), collapse=""), 
           width=plots_grey_6er_width, 
           height=plots_grey_6er_height, 
           units = plots_grey_6er_units,
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", animal,"_", ground , "_grey_en_without_errors_with_midstance.png"), collapse=""), 
           width=plots_grey_6er_width, 
           height=plots_grey_6er_height, 
           units = plots_grey_6er_units,
           dpi = plots_grey_6er_dpi)
  }
}

for(ground in levels(data$ground_type))
{
  for(joint_grp_name in names(joint_grps))
  {
    joint_grp = joint_grps[[joint_grp_name]]
    g <- ggplot(data[(variable %in% plot_vars_4) & ground_type==ground & joint_name%in%joint_grp, ], 
                aes(x=time*100, y=value, color = animal_name, fill=animal_name, group = interaction(animal_name, ground_type, variable, joint_name)))+
      geom_vline(data=midstance_tbl[ground_type==ground,], aes(xintercept=time_midstance*100, color=animal_name), linetype=6, show.legend=FALSE)+
      geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
      geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
      geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
      geom_line(data=data_replaced[ground_type==ground & joint_name%in%joint_grp, ])+
      facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
      scale_color_manual(name="Tier", breaks = animal_breaks, labels=animal_labels, values = qualitative_color_palette)+
      scale_fill_manual(name = "Tier", breaks = animal_breaks, labels=animal_labels, values = qualitative_color_palette)+
      scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = "Rotation (∞)")+
      theme_bw()+
      theme(legend.justification = c(1,0))
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", ground,"_", joint_grp_name , ".pdf"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", ground,"_", joint_grp_name , ".png"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           dpi = standard_dpi)
  }
}

for(ground_grp_idx in seq(ground_grps))
{
  ground_grp = ground_grps[[ground_grp_idx]]
  ground_grp_name = names(ground_grps)[ground_grp_idx]
  
  for(joint_grp_name in names(joint_grps))
  {
    joint_grp = joint_grps[[joint_grp_name]]
    g <- ggplot(interpolations_means[joint_name%in%joint_grp & ground_type%in%ground_grp, ], 
           aes(x=time*100, y=spl_int_mean, color = ground_type, fill=ground_type, group = interaction(ground_type, variable, joint_name)))+
      geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
      geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
      geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
      geom_line()+
      facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4_raw))+
      scale_color_manual(name="Bodentyp", breaks = ground_breaks, labels=ground_labels, values = qualitative_color_palette)+
      scale_fill_manual(name = "Bodentyp", breaks = ground_breaks, labels=ground_labels, values = qualitative_color_palette)+
      scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = "Rotation (∞)")+
      theme_bw()+
      theme(legend.justification=c(1,0))
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", "average","_",ground_grp_name,"_", joint_grp_name , ".pdf"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", "average","_",ground_grp_name,"_", joint_grp_name , ".png"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           dpi = standard_dpi)
  }
  
  
  
  for(joint_grp_name in names(joint_grps))
  {
    joint_grp = joint_grps[[joint_grp_name]]
    g <- ggplot(interpolations_means[joint_name%in%joint_grp & ground_type%in%ground_grp, ], 
                aes(x=time*100, y=spl_int_mean, color = ground_type, fill=ground_type, group = interaction(ground_type, variable, joint_name)))+
      geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
      geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
      geom_line()+
      facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4_raw))+
      scale_color_manual(name="Bodentyp", breaks = ground_breaks, labels=ground_labels, values = qualitative_color_palette)+
      scale_fill_manual(name = "Bodentyp", breaks = ground_breaks, labels=ground_labels, values = qualitative_color_palette)+
      scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = "Rotation (∞)")+
      theme_bw()+
      theme(legend.justification=c(1,0))
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", "average","_",ground_grp_name,"_", joint_grp_name,"_without_ribbons" , ".pdf"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", "average","_",ground_grp_name,"_", joint_grp_name,"_without_ribbons" , ".png"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           dpi = standard_dpi)
  }
  
  
  for(joint_grp_name in names(joint_grps))
  {
    joint_grp = joint_grps[[joint_grp_name]]
    g <- ggplot(interpolations_means[joint_name%in%joint_grp & ground_type%in%ground_grp, ], 
                aes(x=time*100, y=spl_int_mean, color = ground_type, fill=ground_type, group = interaction(ground_type, variable, joint_name)))+
      geom_line()+
      facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4_raw))+
      scale_color_manual(name="Bodentyp", breaks = ground_breaks, labels=ground_labels, values = qualitative_color_palette)+
      scale_fill_manual(name = "Bodentyp", breaks = ground_breaks, labels=ground_labels, values = qualitative_color_palette)+
      scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
      scale_y_continuous(name = "Rotation (∞)")+
      theme_bw()+
      theme(legend.justification=c(1,0))
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", "average","_",ground_grp_name,"_", joint_grp_name,"_without_error_indicators" , ".pdf"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           device=cairo_pdf)
    
    ggsave(g, file=paste(c(plots_dir, "/", "plot_", "average","_",ground_grp_name,"_", joint_grp_name,"_without_error_indicators" , ".png"), collapse=""), 
           width=standard_width, 
           height=standard_height, 
           units = standard_units,
           dpi = standard_dpi)
  }
}

# ggplot(data[variable %in% plot_vars_4, ], 
#        aes(x=time*100, y=value, color = ground_type, fill=ground_type, group = interaction(animal_name, ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   geom_line()+
#   facet_grid(list("variable", c("animal_name", "joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
#   scale_color_manual(values = qualitative_color_palette)+
#   scale_fill_manual(values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (¬∞)")

# ggplot(data[(variable %in% plot_vars_4) & animal_name=="alma", ],
#        aes(x=time*100, y=value, color = ground_type, fill=ground_type, group = interaction(animal_name, ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   geom_line()+
#   facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
#   scale_color_manual(name="Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_fill_manual(name = "Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (∞)")+
#   ggtitle("Rotationen f√ºr Alma auf verschiedenen B√∂den")+
#   theme_bw()+
#   theme(legend.position="top", plot.title = element_text(hjust = 0.5))
# 
# ggplot(data[(variable %in% plot_vars_4) & animal_name=="berta", ], 
#        aes(x=time*100, y=value, color = ground_type, fill=ground_type, group = interaction(animal_name, ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   geom_line()+
#   facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
#   scale_color_manual(name="Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_fill_manual(name = "Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (∞)")+
#   ggtitle("Rotationen f√ºr Berta auf verschiedenen B√∂den")+
#   theme_bw()+
#   theme(legend.position="top", plot.title = element_text(hjust = 0.5))
# 
# ggplot(data[(variable %in% plot_vars_4) & ground_type=="beton", ], 
#        aes(x=time*100, y=value, color = animal_name, fill=animal_name, group = interaction(animal_name, ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   geom_line()+
#   facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
#   scale_color_manual(name="Tier", breaks = c("alma", "berta"), labels=c("Alma", "Berta"), values = qualitative_color_palette)+
#   scale_fill_manual(name = "Tier", breaks = c("alma", "berta"), labels=c("Alma", "Berta"), values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (∞)")+
#   ggtitle("Rotationen auf Beton")+
#   theme_bw()+
#   theme(legend.position="top", plot.title = element_text(hjust = 0.5))
# 
# 
# ggplot(data[(variable %in% plot_vars_4) & ground_type=="kura", ], 
#        aes(x=time*100, y=value, color = animal_name, fill=animal_name, group = interaction(animal_name, ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   geom_line()+
#   facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
#   scale_color_manual(name="Tier", breaks = c("alma", "berta"), labels=c("Alma", "Berta"), values = qualitative_color_palette)+
#   scale_fill_manual(name = "Tier", breaks = c("alma", "berta"), labels=c("Alma", "Berta"), values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (∞)")+
#   ggtitle("Rotationen auf Kura")+
#   theme_bw()+
#   theme(legend.position="top", plot.title = element_text(hjust = 0.5))
# 
# 
# 
# 
# ggplot(data[(variable %in% plot_vars_4) & ground_type=="kura", ], 
#        aes(x=time*100, y=value, color = animal_name, fill=animal_name, group = interaction(animal_name, ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   geom_line(data=data_replaced[ground_type=="kura", ])+
#   facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
#   scale_color_manual(name="Tier", breaks = c("alma", "berta"), labels=c("Alma", "Berta"), values = qualitative_color_palette)+
#   scale_fill_manual(name = "Tier", breaks = c("alma", "berta"), labels=c("Alma", "Berta"), values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (∞)")+
#   ggtitle("Rotationen auf Kura")+
#   theme_bw()+
#   theme(legend.position="top", plot.title = element_text(hjust = 0.5))
# 
# 
# ggplot(data[(variable %in% plot_vars_4) & animal_name=="alma" & (joint_name%in%c("PIP links", "PIP rechts")), ], 
#        aes(x=time*100, y=value, color = ground_type, fill=ground_type, group = interaction(animal_name, ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   geom_line(data=data_replaced[animal_name=="alma" & (joint_name%in%c("PIP links", "PIP rechts")), ])+
#   facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
#   scale_color_manual(name="Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_fill_manual(name = "Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (∞)")+
#   # ggtitle("Rotationen f√ºr Alma auf verschiedenen Bˆden")+
#   theme_bw()+
#   theme(legend.justification = c(0.8,0), #legend.position="top",
#         plot.title = element_text(hjust = 0.5))
# 
# ggplot(data[(variable %in% plot_vars_4) & animal_name=="alma", ], 
#        aes(x=time*100, y=value, color = ground_type, fill=ground_type, group = interaction(animal_name, ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   geom_point(data=data_replaced[animal_name=="alma", ], size=0.005)+
#   facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4))+
#   scale_color_manual(name="Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_fill_manual(name = "Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (¬∞)")+
#   ggtitle("Rotationen f√ºr Alma auf verschiedenen Bˆden")+
#   theme_bw()+
#   theme(legend.position="top", plot.title = element_text(hjust = 0.5))

# ggplot(interpolations_means, 
#        aes(x=time*100, y=spl_int_mean, color = ground_type, fill=ground_type, group = interaction(ground_type, variable, joint_name)))+
#   #geom_hline(yintercept = 0, color = "grey")+
#   #geom_vline(xintercept = c(0,100), color="grey")+
#   geom_line()+
#   geom_ribbon(aes(ymin = value_min_error, ymax = value_max_error), linetype = 2, colour=NA, alpha = 0.2)+
#   geom_line(aes(y=value_min_error), linetype=2, alpha =0.7)+
#   geom_line(aes(y=value_max_error), linetype=2, alpha =0.7)+
#   facet_grid(list("variable", c("joint_name")), scales = "free_y", labeller=labeller(variable = labels4_raw))+
#   scale_color_manual(name="Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_fill_manual(name = "Bodentyp", breaks = c("beton", "kura"), labels=c("Beton", "Kura"), values = qualitative_color_palette)+
#   scale_x_continuous(name = "Zeit (%)", limits = c(0,100), breaks=c(0,20,40,60,80,100))+
#   scale_y_continuous(name = "Rotation (∞)")+
#   #ggtitle("Rotationen gemittelt auf verschiedenen Bˆden")+
#   theme_bw()+
#   theme(legend.justification=c(1,0))#, plot.title = element_text(hjust = 0.5))



