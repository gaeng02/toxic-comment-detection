R = {c_0, c_1, ... , c_i} // a set of centroids -> label, member, radius

``` python
if (||c_i - x||_2 <= eps) :  
    c_i members += x  
    recalculate c_i position  
    recalculate c_i radius  
    if c_i is high enough (calculate overlapping area with others)  
        make new centroid  
```

RAW data : label, centroid_id  
Centroid : label, centroid_id, radius