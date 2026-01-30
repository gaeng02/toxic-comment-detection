
``` python
# S : a set of centroids
# eps_1 : distance threshold (default = 0.5)
# eps_2 : overlap ratio threshold (default = 0.9)

# S : id, vector, members, label, radius
# Data : vector, centroid_id, label

if (min(distance  of input and S) <= eps_1) :   # about same label
    c_i members += x
    recalculate c_i         # mean of members 
    recalculate c_i radius  # the most distant
    if (min(c_i radius, c_j radius) * eps_2 >= ((distance of c_i center and c_j center) - max(c_i radius, c_j radius))) : 
        if (c_i members + c_j members >= 3) : 
            kmeans(c_i + c_j, k) # k wiil be determined by silhoette score. in k=2 and k=3 case
            recalculate c vector and radius
else : 
    x : new c_k
    c_k radius = eps_1

```