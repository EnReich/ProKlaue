# This script reads in data for zones statistics produced
# by the python-R workflow for fitting the imprints with 
# the measurement zones.
# One should specify the level of significance for 
# testing (alpha), the working directory and which data 
# is avaiable.
# This script does some testing, outputs the test 
# tables and some data tables for means.
# Finally it outputs some plots.


library(data.table)
library(ggplot2)
library(viridis)
#library(ggsci)
#library(coefplot)

ALPHA = 0.05
setwd("~/ProKlaue - Benny")
OVERALL_DATA_AVAIABLE = T
ZONES_DATA_AVAIABLE = T
MAX_PRESSURE_DATA_AVAIABLE = T
IN_VIVO_DATA_AVAIABLE = T


# -------------------------------- LABELS, COLORS AND SETTINGS --------------------------


ground_breaks = factor(c("Con", "Kar", "Kur", "proK"), ordered = T)
trial_breaks = factor(c("ExVivo", "InVivoStandStat", "InVivoWalkStat", "InVivoWalkDyn"), ordered = T)


qualitative_color_palette_ground_types <- c(Con = "gray90", Kar="gray70", Kur = "gray50", proK="gray30", Kur_nop = "gray50", Kur_ril="gray30") 
theme_set(theme_bw()+
             theme(axis.text=element_text(size=rel(1.3)),
                   axis.title=element_text(size=rel(1.3)),
                   legend.text = element_text(size=rel(1.3)),
                   legend.title = element_text(size=rel(1.3)),
                   strip.text = element_text(size=rel(1.3)))
          )

standard_width = 297
standard_height = 210
parameter_labeller = as_labeller(c(
  "belflaeche" =  "'contact'~'area'~'['~'cm²'~']'",
  # "druckdurchschn" = "'average'~'pressure'~'['~frac('N','cm'^2)~']'",
  # "druckmax" = "'maximum'~'pressure'~'['~frac('N','cm'^2)~']'",
  "druckdurchschn" = "'average pressure [N/cm²]'",
  "druckmax" = "'maximum'~'pressure'~'['~'N/cm²'~']'",
  "kraftvert" = "'force distribution'",
  "kraftges" = "'force total'",
  "rel_area_loaded" = "'proportion of loaded area'",
  "force_area_loaded" = "'pressure on loaded area'",
  "force_side" = "'pressure distribution'"
), default = label_parsed)

parameter_labeller_abrv = as_labeller(c(
  "belflaeche" =  "A['total']~'['~'cm²'~']'",
  # "druckdurchschn" = "'average'~'pressure'~'['~frac('N','cm'^2)~']'",
  # "druckmax" = "'maximum'~'pressure'~'['~frac('N','cm'^2)~']'",
  "druckdurchschn" = "P['av']~'[N/cm²]'",
  "druckmax" = "P['max']~'['~'N/cm²'~']'",
  "kraftvert" = "'FB'",
  "kraftges" = "'force total'",
  "rel_area_loaded" = "'proportion of loaded area'",
  "force_area_loaded" = "'pressure on loaded area'",
  "force_side" = "'pressure distribution'"
), default = label_parsed)

parameter_labeller_german_abrv = as_labeller(c(
  "belflaeche" =  "A['ges']~'['~'cm²'~']'",
  # "druckdurchschn" = "'average'~'pressure'~'['~frac('N','cm'^2)~']'",
  # "druckmax" = "'maximum'~'pressure'~'['~frac('N','cm'^2)~']'",
  "druckdurchschn" = "P['ges']~'[N/cm²]'",
  "druckmax" = "P['max']~'['~'N/cm²'~']'",
  "kraftvert" = "KV",
  "kraftges" = "kraftges",
  "rel_area_loaded" = "Anteil~der~Belastungsfläche",
  "force_area_loaded" = "Druck~auf~der~Belastungsfläche",
  "force_side" = "Druckverteilung"
), default = label_parsed)


trial_labels = c("ExVivo" =  "Ex Vivo",
                "InVivoStandStat" =  "In Vivo Standing Static",
                "InVivoWalkStat" =  "In Vivo Walking Static",
                "InVivoWalkDyn" =  "In Vivo Walking Dynamic")

trial_labels_ger = c("ExVivo" =  "Ex Vivo",
                     "InVivoStandStat" =  "In Vivo Stand Statisch",
                     "InVivoWalkStat" =  "In Vivo Lauf Statisch",
                     "InVivoWalkDyn" =  "In Vivo Lauf Dynamisch")


trial_labels_abr = c("ExVivo" =  "ExVivo",
                 "InVivoStandStat" =  "InVivoStandStat",
                 "InVivoWalkStat" =  "InVivoWalkStat",
                 "InVivoWalkDyn" =  "InVivoWalkDyn")

trial_labels_abr_ger = c("ExVivo" =  "ExVivo",
                     "InVivoStandStat" =  "InVivoStandStat",
                     "InVivoWalkStat" =  "InVivoLaufStat",
                     "InVivoWalkDyn" =  "InVivoLaufDyn")


trial_labeller = as_labeller(trial_labels)

trial_labeller_ger = as_labeller(trial_labels_ger)



variable_labels = c("boden" = "ground")
variable_labels_german = c("boden" = "Boden")

other_labels_german = c("zones" = "Zonen", "frequency" = "Häufigkeit")


pressure_max_side_labeller = as_labeller(c("lat"="lateral",
                                           "med" = "medial"))

parameter_vec = c("belflaeche", "druckdurchschn", "druckmax")
parameter_vec_zonen = c("rel_area_loaded", "force_area_loaded", "force_side")
parameter_vec_all_trials = c("belflaeche", "druckdurchschn", "druckmax", "kraftvert")
klaue_vec_all_trials = c("lat", "med", "ges")

ground_labels = c("Con" = "Con", "Kar" = "Kar", "Kur" = "Kur", "proK" = "proK")
ground_labels_german = c("Con" = "Bet", "Kar" = "Kar", "Kur" = "Kur", "proK" = "proK")

klaue_labels = c("lat" = "lat", "med" = "med", "ges" = "total")
klaue_labels_german = c("lat" = "lat", "med" = "med", "ges" = "ges")


grounds_for_test = c("Con", "Kar", "Kur", "proK")
grounds_for_in_vivo_plots = c("Con", "Kur")

if(grepl("kraiburg",tolower(getwd()), fixed=TRUE))
{
  grounds_for_test = c("Con", "Kur_nop", "Kur_ril")
  grounds_for_in_vivo_plots = c("Con", "Kur_nop", "Kur_ril")
}


# -------------------------------- HELPER FUNCTIONS --------------------------

GetSiginificantSegmentsAtTop <- function(test_table, 
                                         group_cols, 
                                         group_for_max_cols = group_cols,
                                         xcol1,
                                         xcol2,
                                         max_greater_col = "max_greater",
                                         min_lesser_col = "min_less",
                                         xcol_ordered_levels = levels(test_table[,xcol1, with=FALSE][[1]]),
                                         decision_col,
                                         padding_margin=0.05,
                                         dodge_margin = 0.02,
                                         use_param_max_instead = TRUE,
                                         relative_padding_margin = TRUE,
                                         relative_dodge_margin = TRUE)
{
  group_segments = test_table[(test_table[, decision_col, with = FALSE] == FALSE)[,1],]
  group_segments[, (xcol1) := factor(group_segments[, xcol1, with=FALSE][[1]], levels = xcol_ordered_levels, ordered = TRUE)]
  group_segments[, (xcol2) := factor(group_segments[, xcol2, with=FALSE][[1]], levels = xcol_ordered_levels, ordered = TRUE)]
  group_segments[, xmin := pmin(group_segments[, xcol1, with = FALSE], 
                               group_segments[, xcol2, with = FALSE])]
  group_segments[, xmax := pmax(group_segments[, xcol1, with = FALSE], 
                               group_segments[, xcol2, with = FALSE])]
  
  group_segments[, max_param := lapply(.SD[, max_greater_col, with = FALSE], max), by=group_for_max_cols]
  group_segments[, min_param := lapply(.SD[, min_lesser_col, with = FALSE], min), by=group_for_max_cols]
  
  if(relative_padding_margin)
  {
    if(use_param_max_instead)
    {
      group_segments[,y := max_param+(max_param-min_param)*padding_margin]
    } else
    {
      group_segments[,y := max_greater+(max_param-min_param)*padding_margin]
    }
  }  else
  {
    if(use_param_max_instead)
    {
      group_segments[,y := max_param+padding_margin]
    } else
    {
      group_segments[,y := max_greater+padding_margin]
    }
  }
  

  if(relative_dodge_margin)
  {
    group_segments[,dodge_margin := (max_param-min_param)*dodge_margin]
  }
  else
  {
    group_segments[,dodge_margin := dodge_margin]
  }
    
  setorderv(group_segments, c(group_cols, "y", "xmax", "xmin"),
            order = c(rep(1, length(group_cols)),1,1,-1))
  
  nrow_group_segments = nrow(group_segments)
  group_segments[,idx_row := seq(nrow_group_segments)]
  group_segments[,interaction_group_cols := interaction(group_segments[, group_cols, with=FALSE])]
  
  for(par in levels(group_segments$interaction_group_cols))
  {
    line_y_max = 0.
    group_segments_par = group_segments[interaction_group_cols == par]
    if(nrow(group_segments_par)>0)
    {
      line_y_max_par = max(group_segments_par[,y])
      for(i in seq(nrow(group_segments_par)))
      {
        new_y_min = line_y_max+group_segments_par[i,dodge_margin]
        
        if(group_segments_par[i,y] < new_y_min)
        {
          group_segments[group_segments_par[i,idx_row], y := new_y_min]
        }
        line_y_max = group_segments[interaction_group_cols == par][i,y]
      }
    }
  }
  
  return(group_segments)
  
}


ProduceTestingTable<-function(data_table, 
                    group_cols,
                    pre_group_cols = c(),
                    paired,
                    value_col = "value",
                    test_type="wilcox", 
                    alternative="greater", 
                    presort=mean, 
                    mu = 0,
                    interaction_char = "|")
{
  
  result = FALSE
  
  copy_data_table = copy(data_table)
  
  sort_decreasing = !(alternative %in% c("less", "l"))
  
  group_cols_input <- group_cols
  pre_group_cols_input <- pre_group_cols
  
  if("group" %in% c(group_cols, pre_group_cols))
  {
    data_table[, group.1:= group]
    pre_group_cols[which(pre_group_cols=="group")]="group.1"
    group_cols[which(pre_group_cols=="group")]="group.1"
    if(is.data.table(paired))
    {
      paired[, group.1.1:= group.1]
      paired[, group.1.2:= group.2]
    }
  }
  
  if("pre_group" %in% c(group_cols, pre_group_cols))
  {
    data_table[, pre_group.1:= pre_group]
    pre_group_cols[which(pre_group_cols=="pre_group")]="pre_group.1"
    group_cols[which(pre_group_cols=="pre_group")]="pre_group.1"
    if(is.data.table(paired))
    {
      paired[, pre_group.1:= pre_group]
    }
  }
  
  
  copy_data_table[, pre_group := interaction(copy_data_table[,pre_group_cols,with=FALSE], sep = interaction_char)]
  copy_data_table[, group := interaction(copy_data_table[,c(pre_group_cols, group_cols),with=FALSE], sep = interaction_char)]

  
    if(value_col != "value")
  {
    copy_data_table[, value := copy_data_table[, value_col, with=FALSE]]
  }
  
  presort_bool = FALSE
  if(!is.function(presort))
  {
    if(!presort)
    {
      presort_bool = FALSE
    } else if(presort == "mean")
    {
      presort = mean
      presort_bool = TRUE
    } else if (presort == "first")
    {
      presort = first
      presort_bool = TRUE
    } else
    {
      presort_bool = FALSE
    }
  } else
  {
    presort_bool = TRUE
  }
  
  if(presort_bool)
  {
    ordered_group_levels = copy_data_table[, list(value=presort(value)), by = group][order(value, decreasing = sort_decreasing), group]
    ordered_group_levels = factor(ordered_group_levels, levels = ordered_group_levels, ordered=TRUE)
    copy_data_table[, group := factor(group, levels = ordered_group_levels, ordered = TRUE)]
  }
  
  statistical_measures = copy_data_table[, list(mean=mean(value), 
                                                median = median(value), 
                                                min= min(value), 
                                                max=max(value),
                                                stddev = sqrt(var(value))
                                                ), by = group]
  
  if(presort_bool)
  {
    group_levels = ordered_group_levels
  } else
  {
    group_levels = unique(copy_data_table[, pre_group])
  }
  
  pre_group_levels = unique(copy_data_table[, pre_group])
  
  
  # read in paired groups
  if(is.data.table(paired))
  {
    paired_copy = copy(paired)
    paired_copy[, pre_group := interaction(paired_copy[,pre_group_cols,with=FALSE], sep = interaction_char)]
    paired_copy[, group.1 := interaction(paired_copy[,paste(c(pre_group_cols, group_cols), c("1"), sep = "."), with=FALSE], sep = interaction_char)]
    paired_copy[, group.2 := interaction(paired_copy[,paste(c(pre_group_cols, group_cols), c("1"), sep = "."), with=FALSE], sep = interaction_char)]
  } 
  
  for(pre_group_p in pre_group_levels)
  {
    pre_grouped_data = copy_data_table[pre_group == pre_group_p, ]
    pre_grouped_data_group_levels = sort(unique(pre_grouped_data[, group]))
    if(length(pre_grouped_data_group_levels)>1)
    {
      pre_grouped_data_group_combs = combn(pre_grouped_data_group_levels, 2)
      
      for(i in seq(ncol(pre_grouped_data_group_combs)))
      {
        group_LHS = pre_grouped_data_group_combs[1,i][[1]]
        group_RHS = pre_grouped_data_group_combs[2,i][[1]]
        
        paired_groups = FALSE # whether or not the groups are paired
        paired_grouping_cols = c() # which colomns use to pair up
        if(is.data.table(paired))
        {
          paired_copy_selection = paired_copy[pre_group == pre_group & 
                                                group.1%in%c(group_LHS, group_RHS) &
                                                group.2%in%c(group_LHS, group_RHS)]
          
          if(nrow(paired_copy_selection)>0)
          {
            paired_groups = TRUE
            paired_grouping_cols = paired_copy_selection[, group_cols][[1]]
          } else
          {
            paired_groups = FALSE
          }
        } else if(is.atomic(paired))
        {
          if(is.character(paired))
          {
            if(length(unique(pre_grouped_data[group==group_LHS, paired, with = FALSE])[[1]]) == 
               length(unique(pre_grouped_data[group==group_RHS, paired, with = FALSE])[[1]]))
            {
              if(all(sort(unique(pre_grouped_data[group==group_LHS, paired, with = FALSE])[[1]]) ==
                 sort(unique(pre_grouped_data[group==group_RHS, paired, with = FALSE])[[1]])))
              {
                paired_groups = TRUE
                paired_grouping_cols = c(paired)
              }
            }
          }
        }
        
        if(paired_groups & length(paired_grouping_cols)>0)
        {
          pre_grouped_data_paired = dcast(pre_grouped_data[group%in%unlist(list(group_LHS, group_RHS)),], 
                                          formula = paste("group", paste(paired_grouping_cols, sep = "+"), sep = "~"), 
                                          value.var = "value")
          values_LHS = as.numeric(pre_grouped_data_paired[group == group_LHS, 2:ncol(pre_grouped_data_paired)])
          values_RHS = as.numeric(pre_grouped_data_paired[group == group_RHS, 2:ncol(pre_grouped_data_paired)])
        } else
        {
          values_LHS = as.numeric(pre_grouped_data[group == group_LHS, value])
          values_RHS = as.numeric(pre_grouped_data[group == group_RHS, value])
        }
        
        statistics_LHS = statistical_measures[group == group_LHS, 2:ncol(statistical_measures)]
        colnames(statistics_LHS) = paste(colnames(statistics_LHS), "_LHS", sep = "")
        
        statistics_RHS = statistical_measures[group == group_RHS, 2:ncol(statistical_measures)]
        colnames(statistics_RHS) = paste(colnames(statistics_RHS), "_RHS", sep = "")
        
        if(test_type %in% c("w", "wilcox"))
        {
          test = wilcox.test
        } else if(test_type %in% c("t", "ttest"))
        {
          test = t.test
        }
        
        test_result = test(x=values_LHS, y=values_RHS, paired = paired_groups, mu= mu, alternative = alternative)
        group_LHS_split = strsplit(as.character(group_LHS), split = interaction_char, fixed=TRUE)[[1]]
        group_RHS_split = strsplit(as.character(group_RHS), split = interaction_char, fixed=TRUE)[[1]]
        
        result_new_row = data.table(cbind(matrix(group_LHS_split,nrow=1), 
                               matrix(group_RHS_split[(length(pre_group_cols)+1):(length(pre_group_cols)+length(group_cols))], nrow=1)))
        colnames(result_new_row) = c(pre_group_cols_input, 
                                     paste(group_cols, "LHS", sep = "_"),
                                     paste(group_cols, "RHS", sep = "_"))
        
        result_new_row = cbind(result_new_row, statistics_LHS, statistics_RHS)
        
        result_new_row = cbind(result_new_row, p = test_result$p.value,
                               alternative = alternative,
                               test_type = test_type,
                               paired = paired_groups)
  
        if(!is.data.table(result))
        {
          result = result_new_row 
        } else
        {
          result = rbind(result, result_new_row)
        }
      }
    }
  }
  
  return(result)
}

# -------------------------------- DATA AGGREGATION -----------------------------------


if(!dir.exists("qqplots"))
{
  dir.create("qqplots")
}
if(!dir.exists("plots_overall"))
{
  dir.create("plots_overall")
}
if(!dir.exists("plots_in_vivo"))
{
  dir.create("plots_in_vivo")
}
if(!dir.exists("plots_zones"))
{
  dir.create("plots_zones")
}
if(!dir.exists("test_tables"))
{
  dir.create("test_tables")
}

if(!dir.exists("plots_in_vivo/fill_per_boden"))
{
  dir.create("plots_in_vivo/fill_per_boden")
}
if(!dir.exists("plots_in_vivo/fill_per_boden/single_parameter_and_side"))
{
  dir.create("plots_in_vivo/fill_per_boden/single_parameter_and_side")
}
if(!dir.exists("plots_in_vivo/fill_per_versuch"))
{
  dir.create("plots_in_vivo/fill_per_versuch")
}
if(!dir.exists("plots_in_vivo/fill_per_klaue"))
{
  dir.create("plots_in_vivo/fill_per_klaue")
}
if(!dir.exists("plots_in_vivo/pub"))
{
  dir.create("plots_in_vivo/pub")
}


if(!dir.exists("test_tables"))
{
  dir.create("test_tables")
}

if(OVERALL_DATA_AVAIABLE)
{
  df_means = read.csv("data_means.csv")
  colnames(df_means) <- tolower(colnames(df_means))
  if(length(which(colnames(df_means)=="mess.nr"))>0)
  {
    colnames(df_means)[which(colnames(df_means)=="mess.nr")] <- "messnr"
  }
    
  df_means$messnr <- as.factor(df_means$messnr)
  df_means <- data.table(df_means)
  df_means[boden == "Profi", boden:="proK"]
  df_means[, boden := factor(boden)]
}

if(MAX_PRESSURE_DATA_AVAIABLE)
{
  df_max_pressure = read.csv("data_max_pressure_histo.csv", check.names = FALSE)
  df_max_pressure <- data.table(df_max_pressure)
  df_max_pressure  <- melt(df_max_pressure, measure.vars = as.character(seq(5)), sep="")
  df_max_pressure[, zone:=as.factor(df_max_pressure$variable)]
}

if(IN_VIVO_DATA_AVAIABLE)
{
  df_in_vivo = read.csv("Daten Einzelmessung_InVivo_alle.csv")
  df_in_vivo <- data.table(df_in_vivo[,1:7])
  # df_in_vivo[, versuch_type1:= tstrsplit(gsub("([[:upper:]])", " \\1", versuch), " ")[4]]
  # df_in_vivo[, versuch_type2:= tstrsplit(gsub("([[:upper:]])", " \\1", versuch), " ")[5]]
  df_in_vivo_means = df_in_vivo[, list(value = mean(value, na.rm = TRUE)), 
                                # versuch_type1=first(versuch_type1), 
                                # versuch_type2=first(versuch_type2))
                                by = c("versuch", "parameter", "boden", "klaue", "kuh")]
}

if(ZONES_DATA_AVAIABLE)
{
  df_zones = read.csv("data_zones.csv")
  df_zones <- data.table(df_zones[, 1:7])
}

if(OVERALL_DATA_AVAIABLE && ZONES_DATA_AVAIABLE)
{
  kuh_levels_number_strings <- regmatches(as.character(levels(df_means$kuh)), gregexpr("[[:digit:]]+", as.character(levels(df_means$kuh))))
  kuh_levels_number_strings <- as.numeric(unlist(kuh_levels_number_strings))
  df_means$kuh= factor(df_means$kuh, levels = levels(df_means$kuh)[order(kuh_levels_number_strings)])
  
  
  df_zones$kuh= factor(df_zones$kuh, levels = levels(df_means$kuh))
  df_zones$zone = factor(df_zones$zone)
}

if(OVERALL_DATA_AVAIABLE)
{
  df_means_all_measures <- df_means
  df_means <- df_means[, list(variable=mean(variable)), by = eval(colnames(df_means)[-which(colnames(df_means) %in% c("variable", "Mess.Nr", "messnr"))])]
}

if(ZONES_DATA_AVAIABLE)
{
  df_zones_means = df_zones[, list(value = mean(value)), by = c("parameter", "boden", "kuh", "klaue", "zone")]


  df_zones_means_wide_parameter = dcast(df_zones_means, eval(paste(paste(colnames(df_zones_means)[-which(colnames(df_zones_means) %in% c("value", "parameter"))], collapse="+"), "~ parameter", sep="")), value.var="value")

  df_zones_means_wide_parameter[, rel_area_loaded := area_loaded/area_clipped]


  df_zones_means = melt(df_zones_means_wide_parameter, id.vars = colnames(df_zones_means)[-which(colnames(df_zones_means) %in% c("value", "parameter"))],
                         measure.vars = c("force_side", "rel_area_loaded", "force_area_loaded", "area_loaded"))

  if(colnames(df_zones_means)[5] == "variable")
  {
    colnames(df_zones_means)[5]= "parameter"
  }
  df_zones_means$kuh= factor(df_zones_means$kuh, levels = levels(df_means$kuh))
}


df_means_all_trials = data.table(versuch = factor(), parameter = factor(), boden = factor(), klaue = factor(), kuh = factor (), 
                                 value = numeric())
  
if(IN_VIVO_DATA_AVAIABLE)
{
  df_means_all_trials = rbind(df_means_all_trials, df_in_vivo_means[,list(versuch=versuch, 
                                               parameter = parameter,
                                               boden = boden,
                                               klaue = klaue, 
                                               kuh = kuh, 
                                               value = value)])
}
if(OVERALL_DATA_AVAIABLE)
{
  #join in vivo and ex vivo means
  df_means_all_trials = rbind(df_means_all_trials,
                              df_means[,list(versuch = "ExVivo",
                                            parameter = parameter,
                                            boden=boden,
                                            klaue = "ges",
                                            kuh = kuh,
                                            value = variable)])
}  

ground_breaks = factor(unique(c(as.character(ground_breaks), as.character(levels(df_means_all_trials$boden)))))
trial_breaks = factor(unique(c(as.character(trial_breaks), as.character(levels(df_means_all_trials$versuch)))))
df_means_all_trials[, boden := factor(boden, 
                                      levels = ground_breaks, 
                                      ordered=TRUE)]
df_means_all_trials[, versuch := factor(versuch, 
                                        levels = trial_breaks, 
                                        ordered = TRUE)]


if(OVERALL_DATA_AVAIABLE)
{
  #get wide format of means
  df_means_wide = dcast(df_means, eval(paste(paste(colnames(df_means)[-which(colnames(df_means) %in% c("variable", "boden"))], collapse="+"), "~ boden", sep="")), value.var="variable")
  #df_means_wide = df_means_wide[parameter != "kraftvert", ]
  df_means_wide$parameter = factor(df_means_wide$parameter)
}

if(ZONES_DATA_AVAIABLE)
{
  #get wide format for zones
  df_zones_means_wide = dcast(df_zones_means, eval(paste(paste(colnames(df_zones_means)[-which(colnames(df_zones_means) %in% c("value", "boden"))], collapse="+"), "~ boden", sep="")), value.var="value")
  df_zones_means_wide = df_zones_means_wide[parameter != "kraftvert", ]
  df_zones_means_wide$parameter = factor(df_zones_means_wide$parameter)
  
  df_zones_means_wide_zones = dcast(
    df_zones_means, 
    eval(paste(paste(colnames(df_zones_means)[-which(colnames(df_zones_means) %in% c("value", "zone"))], collapse="+"), "~ zone", sep="")),
    value.var="value")
  df_zones_means_wide_zones = df_zones_means_wide_zones[parameter != "kraftvert", ]
  df_zones_means_wide_zones$parameter = factor(df_zones_means_wide_zones$parameter)
  
  
  
  df_zones_means_over_all_cows = df_zones_means[, list(value=mean(value), var = var(value), stddv = sqrt(var(value))), by = c("parameter", "boden", "klaue", "zone")]
  df_zones_means_over_all_zones = df_zones_means[, list(value=mean(value), var = var(value), stddv = sqrt(var(value))), by = c("parameter", "boden", "kuh")]
  df_zones_means_over_all_cows_and_all_zones = df_zones_means[, list(value=mean(value), var = var(value), stddv = sqrt(var(value))), by = c("parameter", "boden")]
}


# -------------------------------- TESTING -----------------------------------

if(OVERALL_DATA_AVAIABLE)
{
  displacements = data.table(no = 1:1, belflaeche = c(0), druckdurchschn = c(0), druckmax = c(0), kraftvert=c(0))
  displacements_zones = data.table(no = 1:1, area_clipped = c(0), area_loaded = c(0), force_area_loaded = c(0),
                                   force_side = c(0), rel_area_loaded=c(0))
  
  df_shapiro_wil_tests <- data.frame(parameter=factor(), boden=factor(), p=double())
  
  # qq plots for the normality within the groups this is P~boden
  # for(param in levels(df_means[,parameter]))
  # {
  #   for(bod in levels(df_means[,boden]))
  #   {
  #     qqplot <- ggplot(data.frame(variable=df_means[parameter==param & boden==bod, variable]), aes(sample=variable))+stat_qq()
  #     ggsave(plot=qqplot, filename = paste("qqplot_", param, "_", bod, ".pdf", sep=""), path = "qqplots/")
  #     shapiro_test <- shapiro.test(df_means[parameter==param & boden==bod, variable])
  #     
  #     df_shapiro_wil_tests <-  rbind(df_shapiro_wil_tests, data.frame(parameter=param, boden=bod, p = shapiro_test["p.value"]))
  #   }
  #   
  #   qqplot <- ggplot(df_means[parameter==param, c("boden", "variable")], aes(sample=variable, color = boden))+stat_qq()
  #   ggsave(plot=qqplot, filename = paste("qqplot_", param, "_comparison.pdf", sep=""), path = "qqplots/")
  # }
  
  
  
  
  
  test_table_mean_all <- FALSE
  
  for(displacements_row_i in 1:nrow(displacements))
  {
    test_table_mean <- data.table(parameter=factor(), displacement = numeric(), 
                                  boden_less=factor(), boden_greater=factor(), 
                                  n_less = numeric(), n_greater = numeric(),
                                  mean_less =numeric(), mean_greater = numeric(), 
                                  var_less = numeric(), var_greater = numeric(),
                                  min_less = numeric(), max_less = numeric(),
                                  min_greater = numeric(), max_greater = numeric(),
                                  p_t=numeric(), p_wil=numeric())
    for(par in levels(df_means_wide$parameter))
    {
      for(i in 1:(length(levels(df_means$boden))-1))
      {
        for(j in (i+1):length(levels(df_means$boden)))
        {
          boden_i = levels(df_means$boden)[i]
          boden_j = levels(df_means$boden)[j]
          
          vals_i = df_means_wide[parameter==par, get(boden_i)]
          vals_j = df_means_wide[parameter==par, get(boden_j)]
          
          mean_i = mean(vals_i)
          mean_j = mean(vals_j)
          
          boden_g = boden_j
          vals_greater = vals_j
          boden_l = boden_i
          vals_less = vals_i
          if(mean_i>mean_j)
          {
            boden_g = boden_i
            vals_greater = vals_i
            boden_l= boden_j
            vals_less = vals_j
          }
  
          displacement = displacements[displacements_row_i, get(par)]
          
          #ttest
          ttest = t.test(vals_greater, vals_less+displacement, paired=TRUE, alt = "greater")
          
          # if(!is.na(match(0, vals_greater-vals_less)))
          # {
          #   print(par)
          #   print(i)
          #   print(j)
          # }
          # print("i")
          # print(i)
          # print(j)
          #wilcox
          wiltest = wilcox.test(vals_greater, vals_less+displacement, paired=TRUE, alt = "greater")
          
          test_table_mean <- rbind(test_table_mean, data.table(parameter=par, displacement = displacement, 
                                                               boden_less= boden_l, boden_greater =boden_g,
                                                               n_less = length(vals_less), n_greater = length(vals_greater),
                                                               mean_less = mean(vals_less), mean_greater = mean(vals_greater),
                                                               var_less = var(vals_less), var_greater = var(vals_greater),
                                                               min_less = min(vals_less), max_less = max(vals_less),
                                                               min_greater = min(vals_greater), max_greater = max(vals_greater),
                                                               p_t = ttest$p.value, p_wil = wiltest$p.value))
        }
      }
    }
  
    
    test_table_mean[, stddv_less := sqrt(var_less)]
    test_table_mean[, stddv_greater := sqrt(var_greater)]
    test_table_mean = test_table_mean[order(p_t),] 
    test_table_mean[, comparison_t:=test_table_mean[,p_t] > ALPHA/(nrow(test_table_mean)+1-1:nrow(test_table_mean))]
    test_table_mean[, corrected_p_t:=cummax(pmin(test_table_mean[,p_t]*(nrow(test_table_mean)+1-1:nrow(test_table_mean)),1))]
    
    if(length(which(test_table_mean[, comparison_t]))>0)
    {
      k_t = min(which(test_table_mean[, comparison_t]))
    } else
    {
      k_t = nrow(test_table_mean)+1
    }
    test_table_mean[, t_H0_decision := c(rep(FALSE, k_t-1), rep(TRUE, nrow(test_table_mean)-k_t+1))]
    
    test_table_mean = test_table_mean[order(p_wil),] 
    test_table_mean[, comparison_wil:=test_table_mean[,p_wil] > ALPHA/(nrow(test_table_mean)+1-1:nrow(test_table_mean))]
    test_table_mean[, corrected_p_wil:=cummax(pmin(test_table_mean[,p_wil]*(nrow(test_table_mean)+1-1:nrow(test_table_mean)),1))]
    
    
    if(length(which(test_table_mean[, comparison_wil]))>0)
    {
      k_wil = min(which(test_table_mean[, comparison_wil]))
    } else
    {
      k_wil = nrow(test_table_mean)+1
    }
    test_table_mean[, wil_H0_decision := c(rep(FALSE, k_wil-1), rep(TRUE, nrow(test_table_mean)-k_wil+1))]
    
    if(is.data.table(test_table_mean_all))
    {
      test_table_mean_all = rbind(test_table_mean_all, test_table_mean)
    }
    else
    {
      test_table_mean_all = test_table_mean
    }
    
  }
  
  group_segments_test_mean = GetSiginificantSegmentsAtTop(
    test_table = test_table_mean_all, 
    group_cols = c("parameter"), 
    xcol1 = "boden_less", 
    xcol2 = "boden_greater", 
    decision_col = "wil_H0_decision")
}

if(ZONES_DATA_AVAIABLE)
{
  
  test_table_zones_all <- FALSE
  for(displacements_row_i in 1:nrow(displacements))
  {
    test_table_zones <- data.table(parameter=factor(), zone = factor(), klaue = factor(), displacement = numeric(), 
                                   boden_less=factor(), boden_greater=factor(), 
                                   n_less = numeric(), n_greater = numeric(),
                                   mean_less =numeric(), mean_greater = numeric(), 
                                   var_less = numeric(), var_greater = numeric(),
                                   min_less = numeric(), max_less = numeric(),
                                   min_greater =numeric(), max_greater = numeric(),
                                   p_t=numeric(), p_wil=numeric())
    for(par in c("rel_area_loaded", "force_area_loaded"))
    {
      for(zo in levels(df_zones_means$zone))
      {
        for(kl in levels(df_zones_means$klaue))
        {
          for(i in 1:(length(levels(df_zones_means$boden))-1))
          {
            for(j in (i+1):length(levels(df_zones_means$boden)))
            {
              boden_i = levels(df_zones_means$boden)[i]
              boden_j = levels(df_zones_means$boden)[j]
              
              vals_i = df_zones_means_wide[parameter==par & zone == zo & klaue == kl, get(boden_i)]
              vals_j = df_zones_means_wide[parameter==par & zone == zo & klaue == kl, get(boden_j)]
              
              mean_i = mean(vals_i)
              mean_j = mean(vals_j)
              
              boden_g = boden_j
              vals_greater = vals_j
              boden_l = boden_i
              vals_less = vals_i
              if(mean_i>mean_j)
              {
                boden_g = boden_i
                vals_greater = vals_i
                boden_l= boden_j
                vals_less = vals_j
              }
              
              displacement = displacements_zones[displacements_row_i, get(par)]
              
              #ttest
              ttest = t.test(vals_greater, vals_less+displacement, paired=TRUE, alternative = "greater")
              
              # if(!is.na(match(0, vals_greater-vals_less)))
              # {
              #   print(par)
              #   print(i)
              #   print(j)
              # }
              # print("i")
              # print(i)
              # print(j)
              #wilcox
              wiltest = wilcox.test(vals_greater, vals_less+displacement, paired=TRUE, alternative = "greater")
              
              test_table_zones <- rbind(test_table_zones, data.table(parameter=par, zone= zo, 
                                                                     klaue = kl, displacement = displacement, 
                                                                     boden_less= boden_l, boden_greater =boden_g,
                                                                     n_less = length(vals_less), n_greater = length(vals_greater),
                                                                     mean_less = mean(vals_less), mean_greater = mean(vals_greater),
                                                                     var_less = var(vals_less), var_greater = var(vals_greater),
                                                                     min_less = min(vals_less), max_less = max(vals_less),
                                                                     min_greater = min(vals_greater), max_greater = max(vals_greater),
                                                                   p_t = ttest$p.value, p_wil = wiltest$p.value))
            }
          }
        }
      }
    }
    
    for(par in c( "force_side"))
    {
      for(kl in levels(df_zones_means$klaue))
      {
        for(i in 1:(length(levels(df_zones_means$zone))-1))
        {
          for(j in (i+1):length(levels(df_zones_means$zone)))
          {
            boden_i = levels(df_zones_means$zone)[i]
            boden_j = levels(df_zones_means$zone)[j]
            
            vals_i = df_zones_means_wide_zones[parameter==par & klaue == kl, get(boden_i)]
            vals_j = df_zones_means_wide_zones[parameter==par & klaue == kl, get(boden_j)]
            
            mean_i = mean(vals_i)
            mean_j = mean(vals_j)
            
            boden_g = boden_j
            vals_greater = vals_j
            boden_l = boden_i
            vals_less = vals_i
            if(mean_i>mean_j)
            {
              boden_g = boden_i
              vals_greater = vals_i
              boden_l= boden_j
              vals_less = vals_j
            }
            
            displacement = displacements_zones[displacements_row_i, get(par)]
            
            #ttest
            ttest = t.test(vals_greater, vals_less+displacement, paired=TRUE, alt = "greater")
            
            # if(!is.na(match(0, vals_greater-vals_less)))
            # {
            #   print(par)
            #   print(i)
            #   print(j)
            # }
            # print("i")
            # print(i)
            # print(j)
            #wilcox
            wiltest = wilcox.test(vals_greater, vals_less+displacement, paired=TRUE, alt = "greater")
            
            test_table_zones <- rbind(test_table_zones, data.table(parameter=par, zone= "avg", 
                                                                   klaue = kl, displacement = displacement, 
                                                                   boden_less= boden_l, boden_greater =boden_g,
                                                                   n_less = length(vals_less), n_greater = length(vals_greater),
                                                                   mean_less = mean(vals_less), mean_greater = mean(vals_greater),
                                                                   var_less = var(vals_less), var_greater = var(vals_greater),
                                                                   min_less = min(vals_less), max_less = max(vals_less),
                                                                   min_greater = min(vals_greater), max_greater = max(vals_greater),
                                                                   p_t = ttest$p.value, p_wil = wiltest$p.value))
          }
        }
      }
    }
    
    test_table_zones[, stddv_less := sqrt(var_less)]
    test_table_zones[, stddv_greater := sqrt(var_greater)]
    test_table_zones = test_table_zones[order(p_t),] 
    test_table_zones[, comparison_t:=test_table_zones[,p_t] > ALPHA/(nrow(test_table_zones)+1-1:nrow(test_table_zones))]
    test_table_zones[, corrected_p_t:=cummax(pmin(test_table_zones[,p_t]*(nrow(test_table_zones)+1-1:nrow(test_table_zones)),1))]
    
    if(length(which(test_table_zones[, comparison_t]))>0)
    {
      k_t = min(which(test_table_zones[, comparison_t]))
    } else
    {
      k_t = nrow(test_table_zones)+1
    }
    test_table_zones[, t_H0_decision := c(rep(FALSE, k_t-1), rep(TRUE, nrow(test_table_zones)-k_t+1))]
    
    test_table_zones = test_table_zones[order(p_wil),] 
    test_table_zones[, comparison_wil:=test_table_zones[,p_wil] > ALPHA/(nrow(test_table_zones)+1-1:nrow(test_table_zones))]
    test_table_zones[, corrected_p_wil:=cummax(pmin(test_table_zones[,p_wil]*(nrow(test_table_zones)+1-1:nrow(test_table_zones)),1))]
    if(length(which(test_table_zones[, comparison_wil]))>0)
    {
      k_wil = min(which(test_table_zones[, comparison_wil]))
    } else
    {
      k_wil = nrow(test_table_zones)+1
    }
    test_table_zones[, wil_H0_decision := c(rep(FALSE, k_wil-1), rep(TRUE, nrow(test_table_zones)-k_wil+1))]
    
    if(is.data.table(test_table_zones_all))
    {
      test_table_zones_all = rbind(test_table_zones_all, test_table_zones)
    }
    else
    {
      test_table_zones_all = test_table_zones
    }
  }
  
  group_segments_test_zones = GetSiginificantSegmentsAtTop(
    test_table = test_table_zones_all, 
    group_cols = c("parameter", "zone", "klaue"), 
    group_for_max_cols = c("parameter"),
    xcol1 = "boden_less", 
    xcol2 = "boden_greater", 
    xcol_ordered_levels = c("1", "2", "3", "4", "5", as.character(ground_breaks)),   #zones in boden less/greater for average over ground test between zones 
    decision_col = "wil_H0_decision")
  

}

if(OVERALL_DATA_AVAIABLE)
{
  # pairwise differences
  boden_pairs = data.table(parameter = factor(), boden_less=factor(), boden_greater = factor(), 
                           pair_name = factor(), level=numeric(), in_level_order=numeric())
  
  for(par in levels(df_means_wide$parameter))
  {
    means_for_parameter = df_means[parameter==par, list(mean=mean(variable)), by=boden]
    boden_ordered = means_for_parameter[means_for_parameter[, order(mean)],boden]
    
    for(i in 1:(length(boden_ordered)-1))
    {
      boden_less = boden_ordered[i]
      
      for(j in (i+1):length(levels(df_means$boden)))
      {
        boden_greater = boden_ordered[j]
        boden_pairs = rbind(boden_pairs, data.table(parameter = par, boden_less=boden_less, boden_greater=boden_greater,
                                                    pair_name = paste(boden_less, "<", boden_greater, sep=""),
                                                    level = j-i, in_level_order=i))
      }
    }
  }
  
  boden_pairs = boden_pairs[order(parameter, level),]
  
  pair_differences_table = data.table(parameter = factor(), 
                                      pair_name=factor(), 
                                      pair_level=numeric(), 
                                      pair_in_level_order = numeric(),
                                      kuh = factor(), variable = numeric())
  
  for(i in 1:nrow(boden_pairs))
  {
    values = df_means_wide[parameter == boden_pairs[i, parameter], get(as.character(boden_pairs[i, boden_greater]))] - 
      df_means_wide[parameter == boden_pairs[i, parameter], get(as.character(boden_pairs[i, boden_less]))]
  
    values_table = cbind(df_means_wide[parameter == boden_pairs[i, parameter], list(kuh, parameter)], 
                         pair_name = boden_pairs[i, pair_name],
                         pair_level = boden_pairs[i, level],
                         pair_in_level_order = boden_pairs[i, in_level_order],
                         values)
    
    colnames(values_table)[length(colnames(values_table))] <- "variable"
    
    pair_differences_table = rbind(pair_differences_table, values_table)
  }
  
}

if(OVERALL_DATA_AVAIABLE || IN_VIVO_DATA_AVAIABLE)
{
  
  # test complete invivo and exvivo table
  # within grounds
  test_table_all_trials_grounds = ProduceTestingTable(df_means_all_trials[boden %in% grounds_for_test & (!(parameter%in% c("kraftges")))],
                                                         group_cols = c("boden"),
                                                         pre_group_cols = c("versuch", "parameter", "klaue"),
                                                         paired = "kuh")
  
  test_table_all_trials_trial = ProduceTestingTable(df_means_all_trials[boden %in% grounds_for_test & (!(parameter%in% c("kraftges")))],
                                                         group_cols = c("versuch"),
                                                         pre_group_cols = c("boden", "parameter", "klaue"),
                                                         paired = "kuh")
  
  test_table_all_trials_claw = ProduceTestingTable(df_means_all_trials[boden %in% grounds_for_test & klaue !="ges" &
                                                                          (!(parameter%in% c("kraftges")))],
                                                    group_cols = c("klaue"),
                                                    pre_group_cols = c("versuch", "parameter", "boden"),
                                                    paired = "kuh")
  
  
  # reject druckmax for lateral and medial
  test_table_all_trials_grounds = test_table_all_trials_grounds[parameter!="druckmax" | klaue=="ges"]
  test_table_all_trials_trial = test_table_all_trials_trial[parameter!="druckmax" | klaue=="ges"]
  
  # correct p-values
  test_table_all_trials_grounds[, p_adjusted :=p.adjust(p, method = "holm") , by = versuch]
  test_table_all_trials_trial[, p_adjusted :=p.adjust(p, method = "holm") ]
  test_table_all_trials_claw[, p_adjusted :=p.adjust(p, method = "holm") , by = versuch]
  
  # all_p_values = c(test_table_all_trials_grounds[,p], test_table_all_trials_trial[, p])
  # all_p_values_adjusted = p.adjust(all_p_values, method = "holm")
  # test_table_all_trials_grounds[, p_adjusted := all_p_values_adjusted[1:nrow(test_table_all_trials_grounds)]]
  # test_table_all_trials_trial[, p_adjusted := all_p_values_adjusted[nrow(test_table_all_trials_grounds)+ (1:nrow(test_table_all_trials_trial))]]
  
  test_table_all_trials_grounds[, decision := !(p_adjusted<ALPHA)]
  test_table_all_trials_trial[, decision := !(p_adjusted<ALPHA)]
  test_table_all_trials_claw[, decision := !(p_adjusted<ALPHA)]
  
  
  group_segments_test_grounds_for_each_trial = GetSiginificantSegmentsAtTop(
    test_table = test_table_all_trials_grounds, 
    group_cols = c("versuch", "parameter", "klaue"), 
    group_for_max_cols = c("versuch", "parameter", "klaue"),
    xcol1 = "boden_LHS", 
    xcol2 = "boden_RHS", 
    xcol_ordered_levels = ground_breaks,
    decision_col = "decision",
    max_greater_col = "max_LHS",
    min_lesser_col = "min_RHS")
  
  group_segments_test_grounds = GetSiginificantSegmentsAtTop(
    test_table = test_table_all_trials_grounds, 
    group_cols = c("versuch", "parameter", "klaue"), 
    group_for_max_cols = c("parameter", "klaue"),
    xcol1 = "boden_LHS", 
    xcol2 = "boden_RHS", 
    xcol_ordered_levels = ground_breaks,
    decision_col = "decision",
    max_greater_col = "max_LHS",
    min_lesser_col = "min_RHS")
}


# -------------------------------- SAVING TABLES -----------------------------------

# save test tables
if(OVERALL_DATA_AVAIABLE)
{
  setorderv(test_table_mean_all, cols = c("parameter", "displacement", "boden_less"))
  write.table(x = test_table_mean_all, file = "test_tables/test_table_overall.csv", sep = ",", row.names = FALSE)
}

if(ZONES_DATA_AVAIABLE)
{
  setorderv(test_table_zones_all, cols = c("parameter", "displacement", "klaue", "zone", "boden_less", "boden_greater"))
  write.table(x = test_table_zones_all, file = "test_tables/test_table_zones.csv", sep = ",", row.names = FALSE)
}

if(IN_VIVO_DATA_AVAIABLE)
{
  write.table(x = test_table_all_trials_trial, file = "test_tables/test_table_all_trials_per_trial.csv", sep = ",", row.names = FALSE)
  write.table(x = test_table_all_trials_grounds, file = "test_tables/test_table_all_trials_per_ground.csv", sep = ",", row.names = FALSE)
  write.table(x = test_table_all_trials_claw, file = "test_tables/test_table_all_trials_per_claw.csv", sep = ",", row.names = FALSE)
}


# save statistical measurements
if(ZONES_DATA_AVAIABLE)
{
  write.table(x = df_zones_means_over_all_cows, file = "test_tables/means_for_zones_over_cows.csv", sep = ",", row.names = FALSE)
  write.table(x = df_zones_means_over_all_zones, file = "test_tables/means_for_cows_over_zones.csv", sep = ",", row.names = FALSE)
  write.table(x = df_zones_means_over_all_cows_and_all_zones, file = "test_tables/means_for_grounds_over_all_cows_and_all_zones.csv", sep = ",", row.names = FALSE)
}

# -------------------------------- PLOTS -----------------------------------
# ................ PLOTS OVERALL .............
if(OVERALL_DATA_AVAIABLE)
{
  # make and save plots
  plot_overall_lines = ggplot(df_means[parameter %in% parameter_vec, ], 
         aes(x=boden, y=variable, color=kuh, group=interaction(kuh, parameter)))+
    geom_boxplot(aes(group = interaction(parameter, boden), fill = boden), outlier.color = NA, width = 0.3)+
    geom_point()+
    geom_line()+
    facet_wrap(~parameter, scales = "free", labeller = parameter_labeller)+
    scale_color_brewer(palette = "Paired")+
    scale_fill_brewer(palette = "Paired", guide = FALSE)+
    labs(color = "Kuh", fill = "Bodentyp", x="Bodentyp", y="Wert")
  
  ggsave(filename= "plots_overall/lines_with_box.pdf", plot=plot_overall_lines, width = standard_width, height = standard_height, unit="mm")
  ggsave(filename= "plots_overall/lines_with_box.png", plot=plot_overall_lines, width = standard_width, height = standard_height, unit="mm")
  
  overall_boxes_plotter_func <- function(data, segments, labelsGround=FALSE, labelFUNPar = parameter_labeller_abrv, x = "boden", y="variable")
  {
    
    plot = ggplot(data,  
                  aes_string(x=x, y=y, group = "interaction(parameter, boden)"))+
      geom_boxplot(width = 0.7, aes(fill=boden), outlier.shape = 21)
    if(!is.atomic(segments))
    {
      if(nrow(segments)>0)
      {
        segments_data = segments[parameter %in% unique(data[,parameter])]
        plot = plot+geom_segment(data = segments_data, 
                                 mapping=aes(x = xmin, xend = xmax, y=y, yend=y), 
                                 inherit.aes=FALSE)
      }
    }
    
    plot = plot+facet_wrap(~parameter, scales = "free", labeller = labelFUNPar)+
      # scale_color_brewer(palette = "Paired")+
      scale_fill_manual(values = qualitative_color_palette_ground_types, guide = FALSE)+
      theme(axis.title.x=element_blank(),
            axis.title.y=element_blank())
    
    if(first(labelsGround!=FALSE))
    {
      plot = plot+scale_x_discrete(labels = labelsGround)
    }
    
    #labs(color = "Kuh", fill = "Bodentyp", x="Bodentyp", y="Wert"))
    return(plot)
  }
  
  
  width_plot_boxes = 160*2
  height_plot_boxes = 70*2
  
  plot_overall_boxes = overall_boxes_plotter_func(df_means[parameter %in% parameter_vec, ], segments = group_segments_test_mean)
  plot_overall_boxes_ger = overall_boxes_plotter_func(df_means[parameter %in% parameter_vec, ], segments = group_segments_test_mean, 
                                                      labelFUN = parameter_labeller_german_abrv,
                                                      labelsGround=ground_labels_german)
  
  plot_overall_boxes_wo_segs = overall_boxes_plotter_func(df_means[parameter %in% parameter_vec, ], segments = FALSE)
  plot_overall_boxes_wo_segs_ger = overall_boxes_plotter_func(df_means[parameter %in% parameter_vec, ], segments = FALSE, 
                                                      labelFUN = parameter_labeller_german_abrv,
                                                      labelsGround=ground_labels_german)
  
  ggsave(filename= "plots_overall/boxes.pdf", plot=plot_overall_boxes, width = width_plot_boxes , height = height_plot_boxes, unit="mm")
  ggsave(filename= "plots_overall/boxes.png", plot=plot_overall_boxes, width = width_plot_boxes, height = height_plot_boxes, unit="mm")
  
  ggsave(filename= "plots_overall/boxes_ger.pdf", plot=plot_overall_boxes_ger, width = width_plot_boxes , height = height_plot_boxes, unit="mm")
  ggsave(filename= "plots_overall/boxes_ger.png", plot=plot_overall_boxes_ger, width = width_plot_boxes, height = height_plot_boxes, unit="mm")
  
  ggsave(filename= "plots_overall/boxes_without_lines.pdf", plot=plot_overall_boxes_wo_segs, width = width_plot_boxes , height = height_plot_boxes, unit="mm")
  ggsave(filename= "plots_overall/boxes_without_lines.png", plot=plot_overall_boxes_wo_segs, width = width_plot_boxes, height = height_plot_boxes, unit="mm")
  
  ggsave(filename= "plots_overall/boxes_without_lines_ger.pdf", plot=plot_overall_boxes_wo_segs_ger, width = width_plot_boxes , height = height_plot_boxes, unit="mm")
  ggsave(filename= "plots_overall/boxes_without_lines_ger.png", plot=plot_overall_boxes_wo_segs_ger, width = width_plot_boxes, height = height_plot_boxes, unit="mm")
  
  
  for(par in parameter_vec)
  {
    plot_overall_boxes_single = overall_boxes_plotter_func(df_means[parameter ==par, ], segments = group_segments_test_mean)
    
    plot_overall_boxes_single_ger = overall_boxes_plotter_func(df_means[parameter ==par, ], 
                                                               segments = group_segments_test_mean,
                                                               labelFUN = parameter_labeller_german_abrv)
    
    ggsave(filename= paste("plots_overall/boxes_single_", par, ".pdf", sep=""),
           plot=plot_overall_boxes_single,
           height = height_plot_boxes, 
           width = width_plot_boxes/3, 
           unit="mm")
    
    ggsave(filename= paste("plots_overall/boxes_single_", par, ".png", sep=""),
           plot=plot_overall_boxes_single,
           height = height_plot_boxes, 
           width = width_plot_boxes/3, 
           unit="mm")
    
    ggsave(filename= paste("plots_overall/boxes_single_", par, "_ger.pdf", sep=""),
           plot=plot_overall_boxes_single_ger,
           height = height_plot_boxes, 
           width = width_plot_boxes/3, 
           unit="mm")
    
    ggsave(filename= paste("plots_overall/boxes_single_", par, "_ger.png", sep=""),
           plot=plot_overall_boxes_single_ger,
           height = height_plot_boxes, 
           width = width_plot_boxes/3, 
           unit="mm")
  }
  
  
  
  for(par in parameter_vec)
  {
    
    plot_overall_parameter_detailed = ggplot(df_means[parameter == par, ], 
           aes(x=boden, y=variable, group=interaction(kuh, parameter)))+
      geom_boxplot(data = df_means_all_measures[parameter == par, ], 
                   aes(x=boden, y=variable, fill=kuh, group=interaction(kuh, boden, parameter)), 
                   position = "identity", width = 0.1, alpha = 0.3, outlier.color = NA)+
      geom_point(data = df_means_all_measures[parameter == par, ], 
                 aes(x=boden, y=variable, color=kuh),
                 alpha = 0.7)+
      geom_line(aes(color=kuh))+
      scale_color_brewer(palette = "Paired")+
      scale_fill_brewer(palette = "Paired")+
      labs(color = "Kuh", fill = "Kuh", x="Bodentyp", y="Wert")
    
    ggsave(filename= paste("plots_overall/detailed_", par, ".pdf", sep=""), 
           plot=plot_overall_parameter_detailed, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
    ggsave(filename= paste("plots_overall/detailed_", par, ".png", sep=""), 
           plot=plot_overall_parameter_detailed, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
  }
}

# ..... PLOTS PRESSURE MAX ........

# pressure max plots
if(MAX_PRESSURE_DATA_AVAIABLE)
{
  width_plot_pressure_max_hist = 80*2
  height_plot_pressure_max_hist = 60*2
  
  for(sidep in levels(df_max_pressure$side))
  {
    
    plot_pressure_max_single = ggplot(df_max_pressure[side == sidep], 
           aes(x=zone, y=value, fill=boden))+
      geom_col(position=position_dodge(), color="black")+
      facet_grid(.~side, labeller = pressure_max_side_labeller)+
      scale_fill_manual(values = qualitative_color_palette_ground_types)+
      labs(fill = "", x="zones", y="frequency")
    
    plot_pressure_max_single_ger = ggplot(df_max_pressure[side == sidep], 
                                      aes(x=zone, y=value, fill=boden))+
      geom_col(position=position_dodge(), color="black")+
      facet_grid(.~side, labeller = pressure_max_side_labeller)+
      scale_fill_manual(values = qualitative_color_palette_ground_types)+
      labs(fill = "", x=other_labels_german["zones"], y=other_labels_german["frequency"])
    
    
    ggsave(filename= paste("plots_zones/pressure_max_", sidep, ".pdf", sep=""), 
           plot=plot_pressure_max_single, 
           width = width_plot_pressure_max_hist, 
           height = height_plot_pressure_max_hist, 
           unit="mm")
    ggsave(filename= paste("plots_zones/pressure_max_", sidep, ".png", sep=""), 
           plot=plot_pressure_max_single, 
           width = width_plot_pressure_max_hist, 
           height = height_plot_pressure_max_hist, 
           unit="mm")
    
    ggsave(filename= paste("plots_zones/pressure_max_", sidep, "_ger.pdf", sep=""), 
           plot=plot_pressure_max_single_ger, 
           width = width_plot_pressure_max_hist, 
           height = height_plot_pressure_max_hist, 
           unit="mm")
    ggsave(filename= paste("plots_zones/pressure_max_", sidep, "_ger.png", sep=""), 
           plot=plot_pressure_max_single_ger, 
           width = width_plot_pressure_max_hist, 
           height = height_plot_pressure_max_hist, 
           unit="mm")
  }
}

# ..... PLOTS ZONES ........

# zone plots
if(ZONES_DATA_AVAIABLE)
{
  for(par in parameter_vec_zonen)
  {
    plot_zonen_parameter_lines = ggplot(df_zones_means[parameter == par, ], 
         aes(x=boden, y=value, group = interaction(parameter, boden, klaue, zone)))+
    geom_boxplot(position = "identity", width = 0.1, outlier.color = NA)+
    geom_line(aes(group = interaction(parameter, klaue, zone, kuh), color = kuh))+
    geom_point(aes(color = kuh))+
    stat_summary(fun.y=mean, geom="point",
                 fill="white",
                 shape=23, size=2,show.legend = FALSE) +
    facet_grid(klaue~zone)+
    scale_color_brewer(palette = "Paired")+
    scale_fill_brewer(palette = "Paired")
    
    ggsave(filename= paste("plots_zones/detailed_", par,"_with_lines.pdf", sep=""), 
           plot=plot_zonen_parameter_lines, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
    ggsave(filename= paste("plots_zones/detailed_", par,"_with_lines.png", sep=""), 
           plot=plot_zonen_parameter_lines, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
    
    
    plot_zonen_parameter_boxes_colored = ggplot(df_zones_means[parameter == par, ], 
                                        aes(x=boden, y=value, group = interaction(parameter, boden, klaue, zone)))+
      geom_boxplot(position = "identity", width = 0.2, aes(fill = boden), outlier.shape = 21)+
      # geom_point(aes(color = kuh))+
      stat_summary(fun.y=mean, geom="point",
                   fill="white",
                   shape=23, size=2,show.legend = FALSE) +
      facet_grid(klaue~zone)+
      scale_color_brewer(palette = "Paired")+
      scale_fill_brewer(palette = "Paired")
    
    ggsave(filename= paste("plots_zones/", par,"_boxes_colored.pdf", sep=""), 
           plot=plot_zonen_parameter_boxes_colored, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
    ggsave(filename= paste("plots_zones/", par,"_boxes_colored.png", sep=""), 
           plot=plot_zonen_parameter_boxes_colored, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
    
    
    plot_zonen_parameter_boxes = ggplot(df_zones_means[parameter == par, ], 
                                                aes(x=boden, y=value, group = interaction(parameter, boden, klaue, zone)))+
      geom_boxplot(position = "identity", width = 0.7, aes(fill = boden), outlier.shape = 21)+
      geom_segment(data = group_segments_test_zones[parameter == par & zone != "avg", ], 
                   mapping=aes(x = xmin, xend = xmax, y=y, yend=y), 
                   inherit.aes=FALSE)+
      facet_grid(klaue~zone)+
      scale_fill_manual(values = qualitative_color_palette_ground_types, guide = FALSE)+
      theme(axis.title.x=element_blank(),
            axis.title.y=element_blank())
    
    plot_zonen_parameter_boxes_ger = plot_zonen_parameter_boxes+scale_x_discrete(labels=ground_labels_german)
    
    ggsave(filename= paste("plots_zones/", par,"_boxes.pdf", sep=""), 
           plot=plot_zonen_parameter_boxes, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
    ggsave(filename= paste("plots_zones/", par,"_boxes.png", sep=""), 
           plot=plot_zonen_parameter_boxes, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
    
    ggsave(filename= paste("plots_zones/", par,"_boxes_ger.pdf", sep=""), 
           plot=plot_zonen_parameter_boxes_ger, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
    ggsave(filename= paste("plots_zones/", par,"_boxes_ger.png", sep=""), 
           plot=plot_zonen_parameter_boxes_ger, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
  }
  
  plot_zonen_force_side_over_all_grounds_colored = ggplot(df_zones_means[parameter == "force_side"], 
                                      aes(x=zone, y=value, group = interaction(parameter, klaue, zone)))+
    geom_boxplot(position = "identity", width = 0.2, aes(fill=zone), outlier.shape = 21)+
    # geom_boxplot(position = "identity", width = 0.1, aes(fill=boden, 
    #                                                      group =interaction(parameter, klaue, boden, zone)),
    #              outlier.shape = 21,
    #              alpha = 0.5)+
    stat_summary(fun.y=mean, geom="point",
                 fill="white",
                 shape=23, size=3,show.legend = FALSE) +
    facet_grid(klaue~.)+
    scale_color_brewer(palette = "Set3")+
    scale_fill_brewer(palette = "Set3")
  
  ggsave(filename= "plots_zones/force_side_over_all_grounds_colored.pdf", 
         plot=plot_zonen_force_side_over_all_grounds_colored, 
         width = standard_width, 
         height = standard_height, 
         unit="mm")
  ggsave(filename= "plots_zones/force_side_over_all_grounds_colored.png", 
         plot=plot_zonen_force_side_over_all_grounds_colored, 
         width = standard_width, 
         height = standard_height, 
         unit="mm")
  
  
  plot_zonen_force_side_over_all_grounds = ggplot(df_zones_means[parameter == "force_side"], 
                                                  aes(x=zone, y=value, group = interaction(parameter, klaue, zone)))+
    geom_boxplot(position = "identity", width = 0.7, aes(fill=zone), outlier.shape = 21)+
    geom_segment(data = group_segments_test_zones[parameter == "force_side", ], 
                 mapping=aes(x = xmin, xend = xmax, y=y, yend=y), 
                 inherit.aes=FALSE)+
    facet_grid(klaue~.)+
    scale_fill_grey(start = 0.9, end = 0.3)+
    theme(axis.title.x=element_blank(),
          axis.title.y=element_blank())
  
  ggsave(filename= "plots_zones/force_side_over_all_grounds.pdf", 
         plot=plot_zonen_force_side_over_all_grounds, 
         width = standard_height, 
         height = standard_width, 
         unit="mm")
  ggsave(filename= "plots_zones/force_side_over_all_grounds.png", 
         plot=plot_zonen_force_side_over_all_grounds, 
         width = standard_height, 
         height = standard_width, 
         unit="mm")
}

# ..... PLOTS ALL TRIALS ........


if(IN_VIVO_DATA_AVAIABLE)
{
  # plot with x - boden, fill - klaue, facet - versuch
  for(par in parameter_vec_all_trials)
  {
    if(par %in% unique(df_means_all_trials$parameter))
    {
      plot_parameter_over_all_trials_fill_side = ggplot(df_means_all_trials[parameter == par],
                                                        aes(x=boden, y=value))+
        geom_boxplot(width = 0.7, 
                     aes(fill=klaue, group = interaction(parameter, versuch, boden, klaue)), 
                     outlier.shape = 21,
                     position = position_dodge(width = 0.8))+
        facet_wrap(~versuch, scales = "free")+
        # scale_color_brewer(palette = "Paired")+
        # scale_fill_manual(values = qualitative_color_palette_ground_types, guide = FALSE)+
        theme(axis.title.x=element_blank(),
              axis.title.y=element_blank())
      
      ggsave(filename= paste("plots_in_vivo/fill_per_klaue/", par,"_boxes_colored.pdf", sep=""), 
             plot=plot_parameter_over_all_trials_fill_side, 
             width = standard_width, 
             height = standard_height, 
             unit="mm")
    }
  }
  
  
  # plot with x - boden, fill - versuch, facet - klaue
  for(par in parameter_vec_all_trials)
  {
    if(par %in% unique(df_means_all_trials$parameter))
    {
      plot_parameter_over_all_trials_fill_versuch = ggplot(df_means_all_trials[parameter == par],
                                                        aes(x=boden, y=value))+
        geom_boxplot(width = 0.7, 
                     aes(fill=versuch, group = interaction(parameter, versuch, boden, klaue)), 
                     outlier.shape = 21,
                     position = position_dodge(width = 0.8))+
        facet_wrap(~klaue, scales = "free")+
        # scale_color_brewer(palette = "Paired")+
        # scale_fill_manual(values = qualitative_color_palette_ground_types, guide = FALSE)+
        theme(axis.title.x=element_blank(),
              axis.title.y=element_blank())
      
      ggsave(filename= paste("plots_in_vivo/fill_per_versuch/", par,"_boxes_colored.pdf", sep=""), 
             plot=plot_parameter_over_all_trials_fill_versuch, 
             width = standard_width, 
             height = standard_height, 
             unit="mm")
    }
  }
  
  
  # plot with x - versuch, fill - boden, facet - parameter
  for(klaue_p in klaue_vec_all_trials)
  {
    plot_parameter_over_all_trials_fill_ground = ggplot(df_means_all_trials[klaue == klaue_p],
                                                      aes(x=versuch, y=value))+
      geom_boxplot(width = 0.7, 
                   aes(fill=boden, group = interaction(parameter, versuch, boden, klaue)), 
                   outlier.shape = 21,
                   position = position_dodge(width = 0.8))+
      facet_wrap(~parameter, scales = "free")+
      # scale_color_brewer(palette = "Paired")+
      # scale_fill_manual(values = qualitative_color_palette_ground_types, guide = FALSE)+
      theme(axis.title.x=element_blank(),
            axis.title.y=element_blank())
    
    ggsave(filename= paste("plots_in_vivo/fill_per_boden/", klaue_p,"_boxes_colored.pdf", sep=""), 
           plot=plot_parameter_over_all_trials_fill_ground, 
           width = standard_width, 
           height = standard_height, 
           unit="mm")
  }
  
  
  # plot fill per boden single parameter and side
  for(par in parameter_vec_all_trials)
  {
    if(par %in% unique(df_means_all_trials$parameter))
    {
      for(klaue_p in unique(df_means_all_trials[parameter == par & boden %in% grounds_for_in_vivo_plots, klaue]))
      {
        plot_parameter_over_all_trials_fill_ground_single_side  = ggplot(df_means_all_trials[klaue == klaue_p & parameter == par  & boden %in% grounds_for_in_vivo_plots],
                                                            aes(x=versuch, y=value))+
          geom_boxplot(width = 0.7, 
                       aes(fill=boden, group = interaction(parameter, versuch, boden, klaue)), 
                       outlier.shape = 21,
                       position = position_dodge(width = 0.8))+
          scale_fill_manual(values = qualitative_color_palette_ground_types, name = variable_labels["boden"])+
          facet_wrap(~parameter, labeller = parameter_labeller_abrv)+
          scale_x_discrete(labels = trial_labels_abr)+
          theme(axis.title.x=element_blank(),
                axis.title.y=element_blank())
          
          plot_parameter_over_all_trials_fill_ground_single_side_ger  = ggplot(df_means_all_trials[klaue == klaue_p & parameter == par  & boden %in% grounds_for_in_vivo_plots],
                                                                           aes(x=versuch, y=value))+
            geom_boxplot(width = 0.7, 
                         aes(fill=boden, group = interaction(parameter, versuch, boden, klaue)), 
                         outlier.shape = 21,
                         position = position_dodge(width = 0.8))+
            scale_fill_manual(values = qualitative_color_palette_ground_types, name = variable_labels_german["boden"])+
            facet_wrap(~parameter, labeller = parameter_labeller_german_abrv)+
            scale_x_discrete(labels = trial_labels_abr_ger)+
          theme(axis.title.x=element_blank(),
                axis.title.y=element_blank())
        
    
        
        
        ggsave(filename= paste("plots_in_vivo/fill_per_boden/single_parameter_and_side/", par,"_", klaue_p, "_boxes.pdf", sep=""), 
               plot=plot_parameter_over_all_trials_fill_ground_single_side, 
               width = standard_height, 
               height = standard_width, 
               unit="mm")
        
        ggsave(filename= paste("plots_in_vivo/fill_per_boden/single_parameter_and_side/", par,"_", klaue_p, "_boxes.png", sep=""), 
               plot=plot_parameter_over_all_trials_fill_ground_single_side, 
               width = standard_height, 
               height = standard_width, 
               unit="mm")
        
        ggsave(filename= paste("plots_in_vivo/fill_per_boden/single_parameter_and_side/", par,"_", klaue_p, "_boxes_ger.pdf", sep=""), 
               plot=plot_parameter_over_all_trials_fill_ground_single_side_ger, 
               width = standard_height, 
               height = standard_width, 
               unit="mm")
        
        ggsave(filename= paste("plots_in_vivo/fill_per_boden/single_parameter_and_side/", par,"_", klaue_p, "_boxes_ger.png", sep=""), 
               plot=plot_parameter_over_all_trials_fill_ground_single_side_ger, 
               width = standard_height, 
               height = standard_width, 
               unit="mm")
      }
    }
  }
  
  
  width_plot_pub = 160*2 
  width_plot_pub_single = width_plot_pub/3
  height_plot_pub = 70*2
  height_plot_pub_single = height_plot_pub
  
  # publication plots
  for(par in c("druckmax", "kraftvert"))
  {
    if(par %in% unique(df_means_all_trials$parameter))
    {
      for(ver in c("InVivoStandStat", "InVivoWalkStat", "InVivoWalkDyn"))
      {
        if(ver %in% unique(df_means_all_trials$versuch))
        {
          plot_grounds_over_one_trial = ggplot(df_means_all_trials[klaue == "ges" & parameter == par  & boden %in% grounds_for_in_vivo_plots & versuch == ver],
                                                 aes(x=boden, y=value))+
            geom_boxplot(width = 0.7, 
                         aes(fill=boden, group = interaction(parameter, versuch, boden, klaue)), 
                         outlier.shape = 21,
                         position = position_dodge(width = 0.8))+
            scale_fill_manual(values = qualitative_color_palette_ground_types, name = variable_labels["boden"], labels=ground_labels)+
            scale_x_discrete(labels = ground_labels)+
            facet_wrap(~versuch, labeller = trial_labeller)+
            theme(axis.title.x=element_blank(),
                  axis.title.y=element_blank()) 
        
          plot_grounds_over_one_trial_ger = ggplot(df_means_all_trials[klaue == "ges" & parameter == par  & boden %in% grounds_for_in_vivo_plots & versuch == ver],
                                                     aes(x=boden, y=value))+
            geom_boxplot(width = 0.7, 
                         aes(fill=boden, group = interaction(parameter, versuch, boden, klaue)), 
                         outlier.shape = 21,
                         position = position_dodge(width = 0.8))+
            scale_fill_manual(values = qualitative_color_palette_ground_types, name = variable_labels_german["boden"], labels = ground_labels_german)+
            scale_x_discrete(labels = ground_labels_german)+
            facet_wrap(~versuch, labeller = trial_labeller_ger)+
            theme(axis.title.x=element_blank(),
                  axis.title.y=element_blank()) 
          
          
          ggsave(filename= paste("plots_in_vivo/pub/", par,"_", ver, "_boxes_single.pdf", sep=""), 
                 plot=plot_grounds_over_one_trial, 
                 width = width_plot_pub_single , 
                 height = height_plot_pub_single, 
                 unit="mm")
        
          ggsave(filename= paste("plots_in_vivo/pub/", par,"_", ver, "_boxes_single.png", sep=""), 
                 plot=plot_grounds_over_one_trial, 
                 width = width_plot_pub_single , 
                 height = height_plot_pub_single, 
                 unit="mm")
          
          ggsave(filename= paste("plots_in_vivo/pub/", par,"_", ver, "_boxes_single_ger.pdf", sep=""), 
                 plot=plot_grounds_over_one_trial_ger, 
                 width = width_plot_pub_single , 
                 height = height_plot_pub_single, 
                 unit="mm")
          
          ggsave(filename= paste("plots_in_vivo/pub/", par,"_", ver, "_boxes_single_ger.png", sep=""), 
                 plot=plot_grounds_over_one_trial_ger, 
                 width = width_plot_pub_single , 
                 height = height_plot_pub_single, 
                 unit="mm")
        }
        
      }
    }
  }
  
  
  
  for(par in c("druckmax", "belflaeche", "druckdurchschn"))
  {
    if(par %in% levels(df_means_all_trials$parameter))
    {
      for(ver in c("InVivoStandStat", "InVivoWalkStat", "InVivoWalkDyn"))
      {
        if(ver %in% unique(df_means_all_trials$versuch))
        {
          plot_claws_and_grounds_over_one_trial = ggplot(df_means_all_trials[parameter == par  & boden %in% grounds_for_in_vivo_plots & versuch == ver],
                                                 aes(x=klaue, y=value))+
            geom_boxplot(width = 0.7, 
                         aes(fill=boden, group = interaction(parameter, versuch, boden, klaue)), 
                         outlier.shape = 21,
                         position = position_dodge(width = 0.8))+
            scale_fill_manual(values = qualitative_color_palette_ground_types, name = variable_labels["boden"], labels=ground_labels)+
            scale_x_discrete(labels =  klaue_labels)+
            facet_wrap(~versuch, labeller = trial_labeller)+
            theme(axis.title.x=element_blank(),
                  axis.title.y=element_blank()) 
          
          plot_claws_and_grounds_over_one_trial_ger = ggplot(df_means_all_trials[parameter == par  & boden %in% grounds_for_in_vivo_plots & versuch == ver],
                                                         aes(x=klaue, y=value))+
            geom_boxplot(width = 0.7, 
                         aes(fill=boden, group = interaction(parameter, versuch, boden, klaue)), 
                         outlier.shape = 21,
                         position = position_dodge(width = 0.8))+
            scale_fill_manual(values = qualitative_color_palette_ground_types, name = variable_labels_german["boden"], labels=ground_labels_german)+
            scale_x_discrete(labels = klaue_labels_german)+
            facet_wrap(~versuch, labeller = trial_labeller_ger)+
            theme(axis.title.x=element_blank(),
                  axis.title.y=element_blank()) 
          
          
          p_wdt = length(unique(df_means_all_trials[parameter == par  & boden %in% c("Con", "Kur") & versuch == ver, klaue]))*
            width_plot_pub_single
          
          
          ggsave(filename= paste("plots_in_vivo/pub/", par,"_", ver, "_boxes_sides.pdf", sep=""), 
                 plot=plot_claws_and_grounds_over_one_trial, 
                 width = p_wdt, 
                 height = height_plot_pub, 
                 unit="mm")
          
          ggsave(filename= paste("plots_in_vivo/pub/", par,"_", ver, "_boxes_sides.png", sep=""), 
                 plot=plot_claws_and_grounds_over_one_trial, 
                 width = p_wdt, 
                 height = height_plot_pub, 
                 unit="mm")
          
          ggsave(filename= paste("plots_in_vivo/pub/", par,"_", ver, "_boxes_sides_ger.pdf", sep=""), 
                 plot=plot_claws_and_grounds_over_one_trial_ger, 
                 width = p_wdt, 
                 height = height_plot_pub, 
                 unit="mm")
          
          ggsave(filename= paste("plots_in_vivo/pub/", par,"_", ver, "_boxes_sides_ger.png", sep=""), 
                 plot=plot_claws_and_grounds_over_one_trial_ger, 
                 width = p_wdt, 
                 height = height_plot_pub, 
                 unit="mm")
        }
      }
    }
  }
}





# 
# all_trials_boxes_plotter_func <- function(data, segments, x = "boden", y="value")
# {
#   
#   plot = ggplot(data,  
#                 aes_string(x=x, y=y))+
#     geom_boxplot(width = 0.7, aes(fill=versuch), outlier.shape = 21)
#   if(segments!= FALSE)
#   {
#     if(nrow(segments)>0)
#     {
#       segments_data = segments[parameter %in% unique(data[,parameter])]
#       plot = plot+geom_segment(data = segments_data, 
#                                mapping=aes(x = xmin, xend = xmax, y=y, yend=y), 
#                                inherit.aes=FALSE)
#     }
#   }
#   
#   plot = plot+facet_wrap(~parameter, scales = "free", labeller = parameter_labeller)+
#     # scale_color_brewer(palette = "Paired")+
#     # scale_fill_manual(values = qualitative_color_palette_ground_types, guide = FALSE)+
#     theme(axis.title.x=element_blank(),
#           axis.title.y=element_blank())
#   #labs(color = "Kuh", fill = "Bodentyp", x="Bodentyp", y="Wert"))
#   return(plot)
# }
# 




# 
# ggplot(df_zones_means[parameter%in% c("rel_area_loaded", "force_area_loaded")], 
#        aes(x=boden, y=value, color=kuh, group=interaction(kuh, parameter)))+
#   geom_boxplot(aes(group = interaction(parameter, boden)), position = "identity", width = 0.1, alpha = 0.3, outlier.color = NA)+
#   geom_line(aes(linetype = kuh), data = df_zones_means_over_all_zones[parameter%in% c("rel_area_loaded", "force_area_loaded")])+
#   facet_wrap(~parameter, nrow=1, scales = "free")+
#   scale_color_brewer(palette = "Paired")
# 
# 
# ggplot(pair_differences_table, aes(x=pair_name, y = variable))+geom_boxplot()+facet_grid(parameter~pair_level)
# 
# ggplot(df_means[parameter %in% c("belflaeche", "druckdurchschn", "druckmax"), ], 
#        aes(x=boden, y=variable, color=kuh, group=interaction(kuh, parameter)))+
#   geom_line()+
#   facet_grid(parameter~., scales = "free")+
#   scale_color_brewer(palette = "Paired")
# 
# 
# ggplot(df_means[parameter =="druckmax", ], 
#        aes(x=boden, y=variable, color=kuh, group=interaction(kuh, parameter)))+
#   geom_boxplot(data = df_means_all_measures[parameter =="druckmax", ], 
#              aes(x=boden, y=variable, fill=kuh, group=interaction(kuh, boden, parameter)), 
#              position = "identity", width = 0.1, alpha = 0.3, outlier.color = NA)+
#   geom_point(data = df_means_all_measures[parameter =="druckmax", ], 
#                aes(x=boden, y=variable, color=kuh), 
#                alpha = 0.3)+
#   geom_line()+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
# 
# 
# ggplot(df_means[parameter =="druckmax", ], 
#        aes(x=boden, y=variable, color=kuh, group=interaction(kuh, parameter)))+
#   geom_boxplot(data = df_means_all_measures[parameter =="druckmax", ], 
#                aes(x=boden, y=variable, fill=kuh, group=interaction(boden, parameter)), 
#                position = "identity", width = 0.1, alpha = 0.3, outlier.color = NA)+
#   geom_point(data = df_means_all_measures[parameter =="druckmax", ], 
#              aes(x=boden, y=variable, color=kuh), 
#              alpha = 0.3)+
#   geom_line()+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
# 
# 
# ggplot(df_means[parameter%in% c("belflaeche", "druckdurchschn", "druckmax")], 
#        aes(x=boden, y=variable, color=kuh, group=interaction(kuh, parameter)))+
#   geom_boxplot(aes(group = interaction(parameter, boden)), position = "identity", width = 0.1, alpha = 0.3, outlier.color = NA)+
#   geom_line(linetype = "dotdash")+
#   facet_wrap(~parameter, nrow=1, scales = "free")+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
# 
# 
# ggplot(df_means[parameter%in% c("belflaeche", "druckdurchschn", "druckmax")], 
#        aes(x=boden, y=variable, color=kuh, group=interaction(kuh, parameter)))+
#   geom_boxplot(aes(group = interaction(parameter, boden)), position = "identity", width = 0.1, alpha = 0.3, outlier.color = NA)+
#   geom_line(aes(linetype = kuh))+
#   facet_wrap(~parameter, nrow=1, scales = "free")+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
# 
# 
# 
# ggplot(df_zones[parameter =="area_loaded", ], 
#        aes(x=boden, y=value, color=kuh, fill=kuh, group = interaction(parameter, boden, kuh, klaue, zone)))+
#   geom_boxplot(position = "identity", width = 0.1, alpha = 0.3, outlier.color = NA)+
#   geom_point(alpha = 0.3)+
#   geom_line(data=df_zones_means[parameter == "area_loaded"], aes (group = interaction(parameter, kuh, klaue, zone)))+
#   facet_grid(klaue~zone)+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
#   
# 
# 
# ggplot(df_zones[parameter == "area_loaded", ], 
#        aes(x=boden, y=value, fill=boden, group = interaction(parameter, boden, klaue, zone)))+
#   geom_boxplot(position = "identity", width = 0.1, alpha = 0.3)+
#   # geom_line(data=df_zones_means[parameter == "area_loaded", ], aes (group = interaction(parameter, kuh, klaue, zone)))+
#   facet_grid(klaue~zone)+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
# 
# 
# 
# ggplot(df_zones[parameter == "force_area_loaded", ], 
#        aes(x=boden, y=value, fill=boden, group = interaction(parameter, boden, klaue, zone)))+
#   geom_boxplot(position = "identity", width = 0.1, alpha = 0.3)+
#   # geom_line(data=df_zones_means[parameter == "area_loaded", ], aes (group = interaction(parameter, kuh, klaue, zone)))+
#   facet_grid(klaue~zone)+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
# 
# 
# 
# ggplot(df_zones_means[parameter == "force_area_loaded", ], 
#        aes(x=boden, y=value, fill=boden, color=boden, group = interaction(parameter, boden, klaue, zone)))+
#   geom_violin(position = "identity", width = 0.1, alpha = 0.3)+
#   geom_jitter(data = df_zones[parameter == "force_area_loaded", ], width = 0.15, height = 0, alpha = 0.3)+
#   geom_line(aes(group = interaction(parameter, klaue, zone, kuh), color = kuh), alpha=0.3)+
#   facet_grid(klaue~zone)+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)+
#   scale_fill_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
# 
# 
# ggplot(df_zones_means[parameter == "force_area_loaded", ], 
#        aes(x=boden, y=value, fill=boden, color=boden, group = interaction(parameter, boden, klaue, zone)))+
#   geom_violin(position = "identity", width = 0.1, alpha = 0.3)+
#   geom_point(alpha = 0.3)+
#   geom_line(aes(group = interaction(parameter, klaue, zone, kuh), color = kuh), alpha=0.3)+
#   facet_grid(klaue~zone)+
#   scale_color_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)+
#   scale_fill_viridis(option = "inferno", discrete = TRUE, begin = 0, end = 0.75)
# 
# 
# ggplot(df_zones_means[parameter == "force_side", ], 
#        aes(x=boden, y=value, fill=boden, color=boden, group = interaction(parameter, boden, klaue, zone)))+
#   geom_violin(position = "identity", width = 0.1, alpha = 0.3)+
#   geom_point(alpha = 0.3)+
#   geom_line(aes(group = interaction(parameter, klaue, zone, kuh), color = kuh), alpha=0.3)+
#   facet_grid(klaue~zone)+
#   scale_color_viridis(discrete = TRUE, begin = 0, end = 0.7)+
#   scale_fill_viridis(discrete = TRUE, begin = 0, end = 0.7)
# 
# 
# 
# 
# ggplot(df_zones_rel_area[parameter == "rel_area_loaded", ], 
#        aes(x=boden, y=value, fill=boden, color=boden, group = interaction(parameter, boden, klaue, zone)))+
#   geom_boxplot(position = "identity", width = 0.1, alpha = 0.3)+
#   geom_point(alpha = 0.3)+
#   geom_line(aes(group = interaction(parameter, klaue, zone, kuh), color = kuh), alpha=0.3)+
#   stat_summary(fun.y=mean, geom="point", 
#                shape=23, size=2,show.legend = FALSE) +
#   facet_grid(klaue~zone)+
#   scale_color_viridis(discrete = TRUE, begin = 0, end = 0.7)+
#   scale_fill_viridis(discrete = TRUE, begin = 0, end = 0.7)
# 
# # #get wide format for kuh
# # df_means_wide_for_kuh = dcast(df_means, eval(paste(paste(colnames(df_means)[-which(colnames(df_means) %in% c("variable", "kuh"))], collapse="+"), "~ kuh", sep="")), value.var="variable")
# # df_means_wide_for_kuh = df_means_wide_for_kuh[parameter != "kraftvert", ]
# # df_means_wide_for_kuh$parameter = factor(df_means_wide_for_kuh$parameter)
# 
# 
# # ggplot(df_means,aes(x=variable)) + geom_histogram(aes(y=..density..), binwidth=3) + facet_grid(boden ~ parameter, scales="free") + geom_density(col=3)
# 
# 



