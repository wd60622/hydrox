
# Hyrox 

[Hyrox](https://hyrox.com/the-fitness-race/) is a fitness race which involves running and various exercises. 

Inspired by this blog post [here](https://thaddeus-segura.com/hacking-hyrox/).

## Usage 

### Explore Results

![](./images/exercise-times.png)
![](./images/rest-times.png)

### Optimize 

Define time (in minutes) for each exercise and the average pace of your run based on different effort levels. 

```text
                      Maintenance Priority All-In
1000m SkiErg                  NaN      NaN    NaN
50m Sled Push                 NaN      NaN    NaN
50m Sled Pull                 NaN      NaN    NaN
80m Burpee Broad Jump         NaN      NaN    NaN
1000m Row                     NaN      NaN    NaN
200m Farmers Carry            NaN      NaN    NaN
100m Sandbag Lunges           NaN      NaN    NaN
Wall Balls                    NaN      NaN    NaN
Running 1000m                 NaN      NaN    NaN
```

Run the following command to optimize your exercise times.

```bash
python -m hyrox optimize ./your-exercise-times.csv --maintenance 4 --priority 3 --all-in 2
```

The output will look something like this:

```text
The best happens when you do the following effort levels:
Maintenance:
['1000m SkiErg', '50m Sled Push', '50m Sled Pull', '200m Farmers Carry']
Priority:
['80m Burpee Broad Jump', '1000m Row', '100m Sandbag Lunges']
All-In:
['Wall Balls', 'Running 1000m']
```