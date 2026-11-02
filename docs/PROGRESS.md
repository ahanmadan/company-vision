project: Company Vision
version: "1.0"

current_phase: 1
last_approved_node: "1.0"
current_node: null
next_node: "1.1"

completed_nodes: 1
total_nodes: 42

node_1_0:
  title: "Project Scope and Architecture"
  status: PASS
  approved_date: "2026-10-29"
  acceptance_criteria: "8/8"
  evidence: "User-confirmed checks"

node_1_1:
  status: TODO

last_approved_node: "1.2"
next_node: "1.3"
completed_nodes: 3
total_nodes: 42

node_1_2:
  status: PASS
  approved_date: "2026-10-29"
  acceptance_criteria: "8/8"
  evidence: "User-confirmed checks"

Last approved node: 1.3
Current node: 1.4
Current status: IN_PROGRESS
Next node: 2.1
Completed nodes: 4/42


last_approved_node: "1.4"
current_node: null
next_node: "2.1"

completed_nodes: 5
total_nodes: 42

phase_1:
  status: PASS
  completed_nodes: 5

node_1_4:
  status: PASS
  acceptance_criteria: "10/10"
  pytest: "13 passed"
  smoke_test: PASS
  ruff: PASS
  git: CLEAN
  evidence: "User-confirmed tests"

node_2_1:
  status: TODO