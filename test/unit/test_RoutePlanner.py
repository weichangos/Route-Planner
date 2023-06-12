import pandas as pd
from RoutePlanner import locs_to_pd_df, add_augmenting_path_to_graph, create_complete_graph
import pytest
import networkx as nx



def test_locs_to_pd_df_is_a_dataframe(): 
   # Arrange
   mock_data = [['A','B','C'], [(2.0, 2.0), (2.0, 6.0), (0.0, 0.0)]]
   mock_dataframe = pd.DataFrame({'A': [(2.0, 2.0)], 'B': [(2.0, 6.0)], 'C' :[(0.0, 0.0)]})
   mock_type = type(mock_dataframe)
   mock_type_2 = type("mock string")
   # Act
   result = type(locs_to_pd_df(mock_data))
   #Assert
   assert result == mock_type
 
def test_loc_to_pd_df_has_correct_input():
   #Arrange
   valid_input = [['A','B'],[(0,1),(2,3)]]
   invalid_input_keyvalue_error = [{'A':(0,1), 'B':(2,3)}]
   invalid_input_indexvalue_error = ['A','B',(0,1),(2,3)]
   invalid_input_typevalue_error = [123]
   expected_result_valid = pd.DataFrame({'location_id':valid_input[0], 'x':[valid_input[1][i][0] for i in range(len(valid_input[1]))], 'y':[valid_input[1][i][1] for i in range(len(valid_input[1]))]})
   expected_error_message = "Input has to be a list of two lists, first being the nodes and second being the coordinates."

   #Act
   with pytest.raises(ValueError) as error:
      locs_to_pd_df(invalid_input_keyvalue_error)
      locs_to_pd_df(invalid_input_indexvalue_error)
      locs_to_pd_df(invalid_input_typevalue_error)
   result = locs_to_pd_df(valid_input)

   #Assert
   assert str(error.value) == expected_error_message
   assert result.equals(expected_result_valid)


def test_create_complete_graph_has_correct_number_nodes_and_edges():
   #Arrange
   mock_pair_weights = {('a', 'b'): 3, ('a','c'):4, ('b', 'c'):5}
   expected_output = "Graph with 3 nodes and 3 edges"

   #Act
   result = str(create_complete_graph(mock_pair_weights, flip_weights=True))
   
   #Assert
   assert expected_output == result


def test_add_augmenting_path_to_graph_has_correct_length(): 
    #Arrange 
   mock_ori_graph = nx.Graph() 
   mock_ori_graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
   mock_min_weight_edges = [('b', 'c'), ('c', 'd')]
   expected_output = 5
   invalid_output = 4
   #Act
   result = len(add_augmenting_path_to_graph(mock_ori_graph, mock_min_weight_edges).edges) 
   #Assert 
   assert result == expected_output 
   assert result != invalid_output
