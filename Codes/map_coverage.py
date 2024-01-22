import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from skimage import measure


# Create same map everytime so that we can debug easily
np.random.seed(100)


"""
Obstacles are denoted with 1
No obstacle (but already visited region) is denoted with 3
Unexplored area is denoted with 0
Current location of Robot is denoted with 2

"""

"""
Let us take the user input for grid size, number of obstacles and number of agents
and generalize the code instead of using specific grid size and specific number of agents.
"""

print("-------------------------------------")
print("Welcome to the Frontier based exploration!")
print("-------------------------------------")

# Input the size of the occupancy grid from the user
user_size = int(input("Please input the length/breadth of the occupancy grid: "))

# Number of agents input from the user
n_agents = int(input("Enter number of agents: "))

# Create a size of the occupancy grid
grid_size = (user_size,user_size)

# Creating an obstacle grid
obs_map = np.uint8(np.zeros(grid_size))

# Take input from the user to how many obstacles they would like to put in the map
print()
n_obstacles = int(input("Enter how many obstacles in the occupancy grid: "))

for i in range(n_obstacles):
    x = np.random.randint(0,grid_size[0])
    y = np.random.randint(0,grid_size[1])
    print("Obstacle at: ",x," ",y)
    obs_map[x][y] = 1

# print(obs_map.shape)

# Creating mapping colors for empty map, obstacle and robot
mapping_colors = colors.ListedColormap(["black","red","yellow"])


# Creating a coverage map for each agent.
coverage_maps = []
for i in range(n_agents):
    coverage_maps.append(np.uint8(np.zeros(grid_size)))

# Randomly choose two locations for the agents
agents_location = []

while True:
    """
    Place the robots in unique random locations.
    """
    x = np.random.randint(0,grid_size[0])
    y = np.random.randint(0,grid_size[1])
    # Check if there is a robot or the generated random location is obstacle region
    if [x,y] in agents_location or obs_map[x][y] == 1:
        continue
    agents_location.append([x,y])
    # print([x,y])
    if(len(agents_location) == n_agents):
        break   
print("Agents Location: ", agents_location)

# Use this if the robot is struck at centroids (random unexplored location)
random_location = [] 

# Matplotlib animation to view the map exploration
# Create as many figures as number of agents
fig_array = []
viewer_array = []

# To make the figures interactive.
plt.ion()
for i in range(n_agents):
    fig_array.append(plt.figure())
    viewer_array.append(fig_array[-1].add_subplot(111))
    fig_array[-1].show()

def display_map(cov_map,n=None):
    """
    Displays the occupancy grid map for each agent. If n is not supplied,
    it displays the given map directly instead of updating the figures of
    each agent.
    Args:
        cov_map (ndarray): Occupancy grid map
        n (int, optional): Agent id. Defaults to None.
    """
    robot_location = np.uint8(np.zeros(grid_size))
    # Update robot locations in the map
    
    if n is None:
        # Mapping colors for obstacles, unexplored map and agents.
        mapping_colors = colors.ListedColormap(["black","red","yellow"])
        
        for location in agents_location:
            robot_location[location[0]][location[1]] = 2
            
        plt.figure(figsize=grid_size)
        # Combine coverage map and the robot location map
        plt.imshow(plt.np.bitwise_or(cov_map, robot_location), cmap=mapping_colors)

    else:
        # Mapping colors for obstacles, unexplored map and agents.
        mapping_colors = colors.ListedColormap(["black","white","yellow","red"])
        robot_location[agents_location[n][0]][agents_location[n][1]] = 2
        disp_map = cov_map.copy()
        disp_map[agents_location[n][0]][agents_location[n][1]] = 2
        # Update the figure plot of the agent.
        viewer_array[n].clear()
        viewer_array[n].imshow(disp_map, cmap=mapping_colors)
        plt.pause(0.02)
        
display_map(obs_map)

def action(agent_map,loc_x, loc_y, x_add, y_add):
    """Generate an action and search with given location

    Args:
        agent_map (ndarray): Agent map
        loc_x (int): x location of the robot
        loc_y (int): y location of the robot
        x_add (int): action step size
        y_add (int): action step size

    Returns:
        bool: check if the action can be taken and update the location of there is any obstacle around viscinity
    """
    if((loc_x+x_add)<0 or (loc_x+x_add)>grid_size[0]-1 or (loc_y+y_add)<0 or (loc_y+y_add)>grid_size[1]-1
       or [loc_x+x_add, loc_y+y_add] in agents_location):
        return False
    elif obs_map[loc_x+x_add][loc_y+y_add] == 1:
        agent_map[loc_x+x_add][loc_y+y_add] = 1
    else:
        return True

def search_surroundings(agent_map, robot_x, robot_y):
    """
    Search the surroundings and update the occupancy grid of the agent map.

    Args:
        agent_map (ndarray): Occupancy grid map
        robot_x (int): x location of the robot
        robot_y (int): y location of the robot

    Returns:
        ndarray: updated agent occupancy grid map
    """
    if(action(agent_map,robot_x,robot_y,1,0)):
        agent_map[robot_x+1][robot_y] = 3
    if(action(agent_map,robot_x,robot_y,0,1)):
        agent_map[robot_x][robot_y+1] = 3
    if(action(agent_map,robot_x,robot_y,1,1)):
        agent_map[robot_x+1][robot_y+1] = 3
    if(action(agent_map,robot_x,robot_y,-1,0)):
        agent_map[robot_x-1][robot_y] = 3
    if(action(agent_map,robot_x,robot_y,0,-1)):
        agent_map[robot_x][robot_y-1] = 3
    if(action(agent_map,robot_x,robot_y,-1,-1)):
        agent_map[robot_x-1][robot_y-1] = 3
    if(action(agent_map,robot_x,robot_y,-1,1)):
        agent_map[robot_x-1][robot_y+1] = 3
    if(action(agent_map,robot_x,robot_y,1,-1)):
        agent_map[robot_x+1][robot_y-1] = 3
    agent_map[robot_x,robot_y] = 3
    return agent_map

def update_robot_location(current_x, current_y, goal_x, goal_y,agent_map):
    """
    Move the robot according to the gooal location

    Args:
        current_x (int): Current x location of the robot
        current_y (int): Current y location of the robot
        goal_x (int): X location of the goal position
        goal_y (int): Y location of the goal position
        agent_map (ndarray): Map of the agent

    Returns:
        List/Bool: Location if the action is feasible, None if it doesn't
    """
    if(current_x>=goal_x and current_y>=goal_y):
        if(action(agent_map,current_x, current_y, -1, -1)):
            return [current_x-1, current_y-1]
        
        elif (action(agent_map,current_x, current_y, -1, 0)):
            return [current_x-1, current_y]
        
        elif(action(agent_map,current_x, current_y, 0, -1)):
            return [current_x, current_y-1]
        
        elif(action(agent_map,current_x, current_y, 1, -1)):
            return [current_x+1, current_y-1]
        
        elif(action(agent_map,current_x, current_y, -1, 1)):
            return [current_x-1, current_y+1]
        
        elif(action(agent_map,current_x, current_y, 0, 1)):
            return [current_x, current_y+1]
        
        elif(action(agent_map,current_x, current_y, 1, 0)):
            return [current_x+1, current_y]
        
        elif(action(agent_map,current_x, current_y, 1, 1)):
            return [current_x+1, current_y+1]
        
    elif(current_x<=goal_x and current_y>=goal_y):
        if(action(agent_map,current_x, current_y, 1, -1)):
            return [current_x+1, current_y-1]
        
        elif (action(agent_map,current_x, current_y, 1, 0)):
            return [current_x+1, current_y]
        
        elif(action(agent_map,current_x, current_y, 1, 1)):
            return [current_x+1, current_y+1]
        
        elif(action(agent_map,current_x, current_y, 0, -1)):
            return [current_x, current_y-1]
        
        elif(action(agent_map,current_x, current_y, -1, 0)):
            return [current_x-1, current_y-1]
        
        elif(action(agent_map,current_x, current_y, -1, -1)):
            return [current_x-1, current_y-1]
        
        elif(action(agent_map,current_x, current_y, 0, 1)):
            return [current_x, current_y+1]
        
        elif(action(agent_map,current_x, current_y, -1, 1)):
            return [current_x-1, current_y+1]
    
    elif(current_x>=goal_x and current_y<=goal_y):
        if(action(agent_map,current_x, current_y, -1, 1)):
            return [current_x-1, current_y+1]
        
        elif (action(agent_map,current_x, current_y, -1, 0)):
            return [current_x-1, current_y]
        
        elif(action(agent_map,current_x, current_y, -1, -1)):
            return [current_x-1, current_y-1]
        
        elif(action(agent_map,current_x, current_y, 0, 1)):
            return [current_x, current_y+1]
        
        elif(action(agent_map,current_x, current_y, 0, -1)):
            return [current_x, current_y-1]
        
        elif(action(agent_map,current_x, current_y, 1, -1)):
            return [current_x+1, current_y-1]
        
        elif(action(agent_map,current_x, current_y, 1, 0)):
            return [current_x+1, current_y]
        
        elif(action(agent_map,current_x, current_y, 1, 1)):
            return [current_x+1, current_y+1]
        
    elif(current_x<=goal_x and current_y<=goal_y):
        if(action(agent_map,current_x, current_y, 1, 1)):
            return [current_x+1, current_y+1]
        
        elif (action(agent_map,current_x, current_y, 1, 0)):
            return [current_x+1, current_y]
        
        elif(action(agent_map,current_x, current_y, 1, -1)):
            return [current_x+1, current_y-1]
        
        elif(action(agent_map,current_x, current_y, 0, 1)):
            return [current_x, current_y+1]
        
        elif(action(agent_map,current_x, current_y, 0, -1)):
            return [current_x, current_y-1]
        
        elif(action(agent_map,current_x, current_y, -1, -1)):
            return [current_x-1, current_y-1]
        
        elif(action(agent_map,current_x, current_y, -1, 0)):
            return [current_x-1, current_y]
        
        elif(action(agent_map,current_x, current_y, -1, 1)):
            return [current_x-1, current_y+1]
    return None


def share_map():
    """
    Share the map between agents if they are close by.
    """
    for a in range(0,n_agents):
        for b in range(a+1, n_agents):
            d = np.sqrt(np.power((agents_location[a][0] - agents_location[b][0]),2) + np.power((agents_location[a][1] - agents_location[b][1]),2))
            if d<2:
                coverage_maps[a][coverage_maps[b]==3] = 3
                coverage_maps[a][coverage_maps[b]==1] = 1
                coverage_maps[b][coverage_maps[a]==3] = 3
                coverage_maps[b][coverage_maps[a]==1] = 1
        
def choose_random_location(random_location, agnt_loc, agent_map):
    """
    Choose random location from the unexplored grids from the given agent_map.
    This doesn't update becuase we also want to restrict to boids algorithm of the swarm
    Hence, if it is explored or the robot reaches the location, then the new location is
    updated.

    Args:
        random_location (list): random location from the unexplored region of the occupancy grid
        agnt_loc (list): agent location
        agent_map (ndarray): occupancy grid of the map

    Returns:
        1/None: 1 if there are unexplored regions left in the map. Otherwise returns None.
    """
    if((len(random_location)==0) or (agnt_loc == random_location) or (abs(agnt_loc[0]-random_location[0]) + abs(agnt_loc[1]-random_location[1])) < 3):
        random_location.clear()
        indices = np.where(agent_map == 10)
        if(len(indices[0]) > 0):
            x_index = np.random.choice(indices[0])
            y_index = np.random.choice(indices[1])
            random_location.append(x_index)
            random_location.append(y_index)
        else:
            return None
    return 1
def update_coverage():
    """
    Main function that computes the frontier regions and their centroids.
    The nearest frontier is choosen and then given as the goal location to the
    agent. If the centroid of the frontier region is already explored, then choose
    a random location from the unexplored region of the agent.

    Returns:
        Bool: True if the map is not explored completely. Otherwise False.
    """
    for i in range(0,n_agents):
        # For each agent find the centroids of the regions
        agent_map = coverage_maps[i].copy()
        # Change 0 to 10 since region props doesn't consider 0 as a label.
        agent_map[agent_map == 0] = 10
        # Get the labels and the regions
        labels = measure.label(agent_map)
        regions = measure.regionprops(labels)
        # Get the robot x and y locations
        robot_x = agents_location[i][0]
        robot_y = agents_location[i][1]
        # List that stores the frontier centroid distance from the agent location
        centroid_distance = []
        # Dictionary that stores location of the centroid with resepect to the distance.
        frontiers = {}
        # print(len(regions))
        if(len(regions) == 0):
            center = (int(grid_size[0]/2),int(grid_size[1]/2))
            # print("Value at centroid: ",agent_map[int(center[1])][int(center[0])])
            if(agent_map[int(center[1])][int(center[0])]) == 10:
                dist = np.sqrt((np.power((robot_x-int(center[1])),2)+ (np.power((robot_y-int(center[0])),2))))
                if(dist>=2):
                    centroid_distance.append(dist)
                    frontiers[centroid_distance[-1]] = [int(center[1]), int(center[0])]
            # Sort the centroid distances to get the closest
            centroid_distance.sort()
            # If the frontiers exist, search surroundings and move robot towards the centroid
            if len(frontiers)>0:
                closest_fronter = frontiers[centroid_distance[0]]
                agent_map = search_surroundings(agent_map, robot_x, robot_y)
                # print("For agent ", i, " The chosen frontier location is: ", closest_fronter)
                loc = update_robot_location(robot_x,robot_y,closest_fronter[0],closest_fronter[1],agent_map)
                if(loc is not None):
                    # print(loc)
                    agents_location[i] = loc.copy()
                else:
                    print("Robot at Same Location")
            # Else, choose a random location and explore surroundings and move the robot
            else:
                if(choose_random_location(random_location, agents_location[i], agent_map) is not None):
                    print("No Frontiers decided, Choosing Random Unexplored Location",random_location[0]," ",random_location[1])
                    agent_map = search_surroundings(agent_map, robot_x, robot_y)
                    # print("For agent ", i, " The chosen frontier location is: ", closest_fronter)
                    loc = update_robot_location(robot_x,robot_y,random_location[0],random_location[1],agent_map)
                    if(loc is not None):
                        # print(loc)
                        agents_location[i] = loc.copy()
                    else:
                        print("Robot at Same Location")
            # Revert back the label 10 to 0
            agent_map[agent_map == 10] = 0
            coverage_maps[i] = agent_map.copy()
            # Display the each map
            display_map(agent_map,i)
        # Same as above but this one has regions
        else:
            for props in regions:
                center = props.centroid
                # print("Value at centroid: ",agent_map[int(center[1])][int(center[0])])
                if(agent_map[int(center[1])][int(center[0])]) == 10:
                    dist = np.sqrt((np.power((robot_x-int(center[1])),2)+ (np.power((robot_y-int(center[0])),2))))
                    if(dist>1):
                        centroid_distance.append(dist)
                        frontiers[centroid_distance[-1]] = [int(center[1]), int(center[0])]
            centroid_distance.sort()
            if len(frontiers)>0:
                closest_fronter = frontiers[centroid_distance[0]]
                agent_map = search_surroundings(agent_map, robot_x, robot_y)
                # print("For agent ", i, " The chosen frontier location is: ", closest_fronter)
                loc = update_robot_location(robot_x,robot_y,closest_fronter[0],closest_fronter[1],agent_map)
                if(loc is not None):
                    # print(loc)
                    agents_location[i] = loc.copy()
                else:
                    print("Robot at Same Location")
            else:
                
                if(choose_random_location(random_location, agents_location[i], agent_map) is not None):
                    print("No Frontiers decided, Choosing Random Unexplored Location ",random_location[0], " ", random_location[1])
                    agent_map = search_surroundings(agent_map, robot_x, robot_y)
                    # print("For agent ", i, " The chosen frontier location is: ", closest_fronter)
                    loc = update_robot_location(robot_x,robot_y,random_location[0],random_location[1],agent_map)
                    if(loc is not None):
                        # print(loc)
                        agents_location[i] = loc.copy()
                    else:
                        print("Robot at Same Location")
            
            agent_map[agent_map==10] = 0
            coverage_maps[i] = agent_map.copy()
            display_map(agent_map,i)
            
    print("Robot Locations", agents_location)
    # Share the map between the agents
    share_map()
    # Check the agent 1 map for unexplored region
    if(0 in coverage_maps[0]):
        return True
    else:
        return False
# Coutner that counts number of iterations.
counter = 0
while(update_coverage()):
    counter += 1

print("Coverage Finished. Agents: ", n_agents, " Occupancy Grid Size: ",grid_size, " Iterations:  ", counter)

# Display the final map
for i in range(0,n_agents):
  display_map(coverage_maps[i],i)  

plt.ioff()
plt.show()
