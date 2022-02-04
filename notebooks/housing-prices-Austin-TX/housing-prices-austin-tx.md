Predict housing prices in Austin TX with tidymodels and xgboost
================

Original notebook <https://juliasilge.com/blog/austin-housing/>.

# Explore data

``` r
# r
library(tidyverse)
train_raw <- read_csv("train.csv")

train_raw %>%
  count(priceRange)
```

    ## # A tibble: 5 × 2
    ##   priceRange        n
    ##   <chr>         <int>
    ## 1 0-250000       1249
    ## 2 250000-350000  2356
    ## 3 350000-450000  2301
    ## 4 450000-650000  2275
    ## 5 650000+        1819

``` python
# py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_raw = pd.read_csv("train.csv")

train_raw.groupby("priceRange")["priceRange"].count()
```

    ## priceRange
    ## 0-250000         1249
    ## 250000-350000    2356
    ## 350000-450000    2301
    ## 450000-650000    2275
    ## 650000+          1819
    ## Name: priceRange, dtype: int64

## Price distribution

``` r
# r
price_plot <-
  train_raw %>%
  mutate(priceRange = parse_number(priceRange)) %>%
  ggplot(aes(longitude, latitude, z = priceRange)) +
  stat_summary_hex(alpha = 0.8, bins = 50) +
  scale_fill_viridis_c() +
  labs(
    fill = "mean",
    title = "Price"
  )

price_plot
```

![](housing-prices-austin-tx_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` python
# py
def clean_price_range(num): 
  return int(num.split("-")[-1].replace("+",""))

p_df = train_raw.copy()
p_df["priceRange"] = (
  train_raw["priceRange"]
  .apply(lambda x: clean_price_range(x))
  )

xmin = p_df["longitude"].min() - .01
xmax = p_df["longitude"].max() + .01
ymin = p_df["latitude"].min() - .01
ymax = p_df["latitude"].max() + .01

plt.cla()
fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(8, 5))

hb = plt.hexbin(
  p_df["longitude"], p_df["latitude"], C=p_df["priceRange"], 
  gridsize=50, cmap='viridis', 
  alpha=.8
  )
_ = ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
_ = ax.set_title("Price")
cb = fig.colorbar(hb, ax=ax)

# plt.show()
```

<img src="housing-prices-austin-tx_files/figure-gfm/unnamed-chunk-4-1.png" width="768" />

## More information with charts

``` r
# r
plot_austin <- function(var, title) {
  train_raw %>%
    ggplot(aes(longitude, latitude, z = {{ var }})) +
    stat_summary_hex(alpha = 0.8, bins = 50) +
    scale_fill_viridis_c() +
    labs(
      fill = "mean",
      title = title
    )
}

library(patchwork)
(price_plot + plot_austin(avgSchoolRating, "School rating")) /
  (plot_austin(yearBuilt, "Year built") + plot_austin(log(lotSizeSqFt), "Lot size (log)"))
```

![](housing-prices-austin-tx_files/figure-gfm/unnamed-chunk-5-3.png)<!-- -->

``` python
# py
xmin = p_df["longitude"].min() - .01
xmax = p_df["longitude"].max() + .01
ymin = p_df["latitude"].min() - .01
ymax = p_df["latitude"].max() + .01

plt.cla()
fig, axs = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(8, 5))
fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

col_names = ["priceRange","avgSchoolRating","yearBuilt","lotSizeSqFt"]
col_pos = 0
for col in range(2): 
  for row in range(2):
    ax = axs[col,row]
    hb = ax.hexbin(
      p_df["longitude"], p_df["latitude"], C=p_df[col_names[col_pos]], 
      gridsize=50, cmap='viridis', 
      alpha=.8
      )
    _ = ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    _ = ax.set_title(col_names[col_pos])
    cb = fig.colorbar(hb, ax=ax)
    col_pos += 1

# plt.show()
```

<img src="housing-prices-austin-tx_files/figure-gfm/unnamed-chunk-6-1.png" width="768" />
