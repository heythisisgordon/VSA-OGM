# Application of VSA-OGM to Pre-generated Occupancy Grids

## Combined Input Processing:

- Use densified 2D occupancy grid to identify free and occupied spaces
- Convert into a labeled point cloud format where each point has coordinates and a binary label (occupied/free)

# VSA-OGM Processing:

- Feed this labeled point cloud into VSA-OGM
- Sample regular points in unoccupied cells in the 2D point cloud for analysis
- Take advantage of VSA-OGM's probabilistic modeling to handle uncertainties and conflicts between your input sources
- Adjust the length scale parameter to match the desired resolution

# Enhanced Output:

- VSA-OGM will generate probability and entropy maps
- These maps can provide a more nuanced representation than binary occupancy
- You'll get additional information about uncertainty in different regions