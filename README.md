# Route-Planner
**How it works:**

Step 1: Generate graph for the input 

Step 2: Find all nodes that have an odd number of edges attached 

Step 3: Find all possible sets of pairings of these nodes and their minimum distances, and sum up these distances for each set 

Step 4: Find the set with the least total distance 

Step 5: Integrate the set back into the original graph and find the result 

Ref video: https://www.youtube.com/watch?v=JCSmxUO0v3k


**How to run RoutePlanner.py:**
- Input the location names and their correspinding coordinates in parametre 'locs', as a list containing two lists. The first list would be the location names and the second the coordinates. 

e.g. locs = [['location_A','loaction_B','location_C'],[(0,0), (1,1), (2,2)]]

- Output is logged in logging.log
