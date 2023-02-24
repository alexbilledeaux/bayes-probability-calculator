import bayes

dataSet = bayes.importDataset("test.data")

# Prior Probability Test Suite
assert round(bayes.getPriorProbability('Yes', dataSet), 3) == 0.6, f"Expected 0.6 for P(Yes). Recieved {round(bayes.getPriorProbability('Yes', dataSet), 3)}."
assert round(bayes.getPriorProbability('No', dataSet), 3) == 0.4, f"Expected 0.6 for P(Yes). Recieved {bayes.getPriorProbability('No', dataSet)}."

# Likelihood Test Suite
assert round(bayes.getLikelihood('Attribute', 'A', 'Yes', dataSet), 3) == 0.667, f"Expected 0.667 for P(A|Yes). Recieved {round(bayes.getLikelihood('Attribute', 'A', 'Yes', dataSet), 3)}."
assert round(bayes.getLikelihood('Attribute', 'B', 'Yes', dataSet), 3) == 0.333, f"Expected 0.333 for P(B|Yes). Recieved {round(bayes.getLikelihood('Attribute', 'B', 'Yes', dataSet), 3)}."
assert round(bayes.getLikelihood('Attribute', 'A', 'No', dataSet), 3) == 0.5, f"Expected 0.5 for P(A|No). Recieved {round(bayes.getLikelihood('Attribute', 'A', 'No', dataSet), 3)}."
assert round(bayes.getLikelihood('Attribute', 'B', 'No', dataSet), 3) == 0.5, f"Expected 0.5 for P(B|No). Recieved {round(bayes.getLikelihood('Attribute', 'B', 'No', dataSet), 3)}."

# Evidence Test Suite
assert round(bayes.getEvidence('Attribute', 'A', dataSet), 3) == 0.6, f"Expected 0.6 for P(A). Recieved {round(bayes.getEvidence('Attribute', 'A', dataSet), 3)}."
assert round(bayes.getEvidence('Attribute', 'B', dataSet), 3) == 0.4, f"Expected 0.6 for P(B). Recieved {round(bayes.getEvidence('Attribute', 'B', dataSet), 3)}."

# Likelihood Test Suite
assert round(bayes.getPosteriorProbability('Yes','Attribute', 'A', dataSet), 3) == 0.667, f"Expected 0.667 for P(Yes|A). Recieved {round(bayes.getPosteriorProbability('Yes','Attribute', 'A', dataSet), 3)}."
assert round(bayes.getPosteriorProbability('Yes','Attribute', 'B', dataSet), 3) == 0.5, f"Expected 0.333 for P(Yes|B). Recieved {round(bayes.getPosteriorProbability('Yes','Attribute', 'B', dataSet), 3)}."
assert round(bayes.getPosteriorProbability('No','Attribute', 'A', dataSet), 3) == 0.333, f"Expected 0.5 for P(No|A). Recieved {round(bayes.getPosteriorProbability('No','Attribute', 'A', dataSet), 3)}."
assert round(bayes.getPosteriorProbability('No','Attribute', 'B', dataSet), 3) == 0.5, f"Expected 0.5 for P(No|B). Recieved {round(bayes.getPosteriorProbability('No','Attribute', 'B', dataSet), 3)}."

print("All tests successful!")