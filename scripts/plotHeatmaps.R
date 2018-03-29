library(ggplot2)
library(akima)
library(viridis)
library(ggthemes)
library(plyr)
library(data.table)
library(pracma)


setwd("~/ProKlaue/testdaten/heatmaps/daten")


resX = resY = 0.01
decDigits = 2

createHeatmap <- function(df, binwidth = 0.5, x="x", y="y", z="z", tiles = FALSE, contour = TRUE)
{
  g = ggplot(df) + aes_string(x=x, y=y, z=z, fill=z)
  if(!tiles)
  {
    g = g + geom_raster() 
  }
  else
  {
    g = g + geom_tile(width=resX, height=resY) 
  }
  if(contour)
  {
    g = g + stat_contour(color = "white", size=0.1, binwidth = binwidth)
  }
  
  g = g+ scale_fill_viridis(na.value= NA)+
          theme_bw()
  
  return(g)
}

all_data=list()
all_plots = list()

for(f in list.files(pattern="*.plane$"))
{
  data_plane = read.csv(f, sep="\t")
  # data_plane.summary = ddply(data_plane, .(N), summarize, Tx.mean=mean(Tx), Ty.mean=mean(Ty), Tz.mean=mean(Tz))
  data_plane.summary = data.table(data_plane)[, list(Tx.mean=mean(Tx), Ty.mean=mean(Ty), Tz.mean=mean(Tz)), by = N]
  plane_p0 = as.matrix(as.numeric(data_plane.summary[1, -1]))
  plane_p1 = as.matrix(as.numeric(data_plane.summary[2, -1]))
  plane_p2 = as.matrix(as.numeric(data_plane.summary[3, -1]))

  x_plane = plane_p1-plane_p0
  y_plane = plane_p2-plane_p0
  x_plane = x_plane /norm(x_plane)
  y_plane = y_plane /norm(y_plane)
  
  n_plane = cross(plane_p1-plane_p0, plane_p2-plane_p0)
  n_plane = n_plane / norm(n_plane)

  if(abs(dot(y_plane, x_plane))>1e-06)
  {
    y_plane = cross(n_plane, x_plane)
  }
  
  name = strsplit(f, ".", fixed=TRUE)[[1]][1]
  all_data[[name]]=list(data_plane =  data_plane.summary, n_plane = n_plane, x_plane = x_plane, y_plane =y_plane, plane_p0 = plane_p0)
}

for(f in list.files(pattern="*.map$"))
{
  print(f)
  
  name = strsplit(f, ".", fixed=TRUE)[[1]][1]
  
  data = read.csv(f, sep="\t")
  
  # data.summary = ddply(data, .(N), summarize, Tx.mean=mean(Tx), Ty.mean=mean(Ty), Tz.mean=mean(Tz), d.mean =mean(d), Tx.sd = sd(Tx), Ty.sd= sd(Ty), Tz.sd = sd(Tz), d.sd = sd(d), sample.size = length(d))
  data.summary = data.table(data)[, list(Tx.mean=mean(Tx), Ty.mean=mean(Ty), Tz.mean=mean(Tz), d.mean =mean(d), Tx.sd = sd(Tx), Ty.sd= sd(Ty), Tz.sd = sd(Tz), d.sd = sd(d), sample.size = length(d)), by = N]
  
  plane_p0 = all_data[[name]]$plane_p0
  x_plane =  all_data[[name]]$x_plane
  y_plane =  all_data[[name]]$y_plane
  
  if(!is.null(plane_p0))
  {
    
    xproj = (data.summary$Tx.mean-plane_p0[1])*x_plane[1]+
      (data.summary$Ty.mean-plane_p0[2])*x_plane[2]+
      (data.summary$Tz.mean-plane_p0[3])*x_plane[3]
    
    yproj = (data.summary$Tx.mean-plane_p0[1])*y_plane[1]+
      (data.summary$Ty.mean-plane_p0[2])*y_plane[2]+
      (data.summary$Tz.mean-plane_p0[3])*y_plane[3]
    
    data.summary$x.proj = xproj
    data.summary$y.proj = yproj    
  
    # for(i in seq(nrow(data.summary)))
    # {
    #   p = as.matrix(as.numeric(data.summary[i, c("Tx.mean", "Ty.mean", "Tz.mean")]))
    #   data.summary[i, "x.proj"] = dot((p-plane_p0) , x_plane)
    #   data.summary[i, "y.proj"] = dot((p-plane_p0) , y_plane)
    # }
    

    
    x_start = floor(min(data.summary$x.proj)*(1/resX))*resX
    x_end = ceil(max(data.summary$x.proj)*(1/resX))*resX
    y_start = floor(min(data.summary$y.proj)*(1/resY))*resY
    y_end = ceil(max(data.summary$y.proj)*(1/resY))*resY
    
    di.plane <- interp(data.summary$x.proj, data.summary$y.proj, data.summary$d.mean,
                 xo=round(seq(x_start, x_end, by = resX), decDigits),
                 yo=round(seq(y_start, y_end, by = resY), decDigits))
    
    data.interp.plane <- interp2xyz(di.plane)
    
    all_data[[name]]$di.plane = di.plane
    all_data[[name]]$data.interp.plane = data.interp.plane
  }
  
  x_start = floor(min(data.summary$Tx.mean)*(1/resX))*resX
  x_end = ceil(max(data.summary$Tx.mean)*(1/resX))*resX
  y_start = floor(min(data.summary$Tz.mean)*(1/resY))*resY
  y_end = ceil(max(data.summary$Tz.mean)*(1/resY))*resY
  
  di.global <- interp(data.summary$Tx.mean, data.summary$Tz.mean, data.summary$d.mean,
                      xo=round(seq(x_start, x_end, by = resX), decDigits),
                      yo=round(seq(y_start, y_end, by = resY), decDigits))
  
  data.interp.global <- interp2xyz(di.global)
  

  all_data[[name]]$data.summary = data.summary
  all_data[[name]]$di.global = di.global
  all_data[[name]]$data.interp.global = data.interp.global
}


plots=list()
data_in_one = data.table()
data_in_one_tris = data.table()
for (name in names(all_data))
{
  split_name = strsplit(name, "_", fixed=TRUE)[[1]]
  idx = as.numeric(split_name[1])
  bone = split_name[2]
  side = split_name[3]
  
  ndf = data.table(all_data[[name]]$data.interp.plane)
  ndf[, c("idx", "bone", "side", "type") := list(idx, factor(bone), factor(side), factor("plane"))]
  data_in_one = rbind(data_in_one, ndf)

  ndf = data.table(all_data[[name]]$data.interp.global)
  ndf[, c("idx", "bone", "side", "type") := list(idx, factor(bone), factor(side), factor("global"))]
  data_in_one = rbind(data_in_one, ndf)
  
  ndf = data.table(all_data[[name]]$data.summary)
  ndf[, c("idx", "bone", "side") := list(idx, factor(bone), factor(side))]
  data_in_one_tris = rbind(data_in_one_tris, ndf)

  plots[[name]]$plane=createHeatmap(as.data.frame(all_data[[name]]$data.interp.plane))
  plots[[name]]$global=createHeatmap(as.data.frame(all_data[[name]]$data.interp.global))
}


# createHeatmap(data_in_one[data_in_one$type == "plane", ])+facet_grid(idx ~ side)

avg_field = data.table(data_in_one)
grp = c("type", "bone", "side", "x", "y")
#setkeyv(avg_field, cols=grp)
x = avg_field[, list(z.mean=mean(z), z.sd = sd(z), sample.size = length(z)), by = grp]

avg_field = data.table(data_in_one_tris)
grp = c("bone", "side", "N")
#setkeyv(avg_field, cols=grp)
t = avg_field[, list(Tx.mean.mean = mean(Tx.mean), Tx.mean.sd = sd(Tx.mean), Ty.mean.mean = mean(Ty.mean), Ty.mean.sd = sd(Ty.mean), Tz.mean.mean = mean(Tz.mean), Tz.mean.sd = sd(Tz.mean), x.proj.mean = mean(x.proj), x.proj.sd = sd(x.proj), y.proj.mean = mean(y.proj), y.proj.sd = sd(y.proj), d.mean.mean=mean(d.mean), d.mean.sd = sd(d.mean), sample.size = length(d.mean)), by = grp]
x_start = floor(min(t[, "x.proj.mean"])*(1/resX))*resX
x_end = ceil(max(t[, "x.proj.mean"])*(1/resX))*resX
y_start = floor(min(t[, "y.proj.mean"])*(1/resY))*resY
y_end = ceil(max(t[, "y.proj.mean"])*(1/resY))*resY
di.t.li <- interp(as.numeric(as.matrix(t[side == "li", "x.proj.mean"])), 
               as.numeric(as.matrix(t[side == "li", "y.proj.mean"])),
               as.numeric(as.matrix(t[side == "li", "d.mean.mean"])),
                    xo=round(seq(x_start, x_end, by = resX), decDigits),
                    yo=round(seq(y_start, y_end, by = resY), decDigits))
t.li.interp <- interp2xyz(di.t.li)
