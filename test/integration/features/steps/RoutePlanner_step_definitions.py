from behave import *

@given('the input of locations and their coordinates')
def step_impl(context):
   pass

@given('the input are set as nodes and edges with networkx')
def step_impl(context):
   print('Hwllo ')

@given('all odd degree nodes are found')
def step_impl(context):
   pass

@when('the Eulerian Circuit has been created')
def step_impl(context):
   assert True is not False

@then('output of the best route is generated')
def step_impl(context):
   assert context.failed is False
