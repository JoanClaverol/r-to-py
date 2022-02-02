Predict ratings for \#TidyTuesday board games
================

# Explore data

### Distribution average ratings

``` r
library(tidyverse, quietly = TRUE)

ratings <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/ratings.csv", col_types = cols())
details <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/details.csv", col_types = cols())

ratings_joined <-
  ratings %>%
  left_join(details, by = "id")

ggplot(ratings_joined, aes(average)) +
  geom_histogram(alpha = 0.8)
```

![](ratings-board-games_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

``` python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/ratings.csv")
details = pd.read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/details.csv")

ratings_joined = (
  ratings
  .join(details, on="id", how="left", lsuffix='_left', rsuffix='_right')
  .copy()
  )


sns.histplot(data=ratings_joined, x="average", alpha=.8)
```

<img src="ratings-board-games_files/figure-gfm/unnamed-chunk-2-1.png" width="768" />

### Distribution minimum age

``` r
ratings_joined %>%
  filter(!is.na(minage)) %>%
  mutate(minage = cut_number(minage, 4)) %>%
  ggplot(aes(minage, average, fill = minage)) +
  geom_boxplot(alpha = 0.2, show.legend = FALSE)
```

![](ratings-board-games_files/figure-gfm/unnamed-chunk-3-3.png)<!-- -->

``` python
p_df = (
ratings_joined
  [ratings_joined["minage"].notnull()]
  .assign(minage = lambda x: pd.cut(x["minage"], bins=4))
)

g = sns.boxplot(data=p_df, x="minage", y="average", hue="minage")
plt.legend([], [], frameon=False)
```

<img src="ratings-board-games_files/figure-gfm/unnamed-chunk-4-3.png" width="768" />
