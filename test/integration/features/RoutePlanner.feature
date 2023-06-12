Feature: theme tests

    Scenario Outline: Route Planner
        Given the input of locations and their coordinates
        And the input are set as nodes and edges with networkx
        And all odd degree nodes are found
        When the Eulerian Circuit has been created
        Then output of the best route is generated
    
