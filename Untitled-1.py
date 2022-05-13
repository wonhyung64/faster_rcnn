#%%
! pip install neptune-client
import neptune.new as neptune

#%%
run = neptune.init(project="model-frcnn", 
                   api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZTNlODVlMi0xMzIyLTQwYzQtYmNkYy1kNWYyZmM1MGFiMjcifQ==")
#%%
params = {"learning_rate": 0.1}

# log params
run["parameters"] = params

# log name and append tags
run["sys/name"] = "basic-colab-example"
run["sys/tags"].add(["colab", "intro"])

# log loss during training
for epoch in range(100):
    run["train/loss"].log(0.99 ** epoch)

# log train and validation scores
run["train/accuracy"] = 0.95
run["valid/accuracy"] = 0.93

params = {"learning_rate": 0.09}

# log params
run["parameters"] = params

# log name and append tags
run["sys/name"] = "example2"
run["sys/tags"].add(["hello", "world"])

# log loss during training
for epoch in range(100):
    run["train/loss"].log(0.5 ** epoch)

# log train and validation scores
run["train/accuracy"] = 0.8
run["valid/accuracy"] = 0.4