#%%
import neptune.new as neptune
import os
import subprocess

#%%
token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZTNlODVlMi0xMzIyLTQwYzQtYmNkYy1kNWYyZmM1MGFiMjcifQ=="
project = "wonhyung64/frcnn-model"
os.environ["NEPTUNE_API_TOKEN"] = token

run = neptune.init(project=project, 
                    api_token=token,
                   mode="offline")

params = {"learning_rate": 0.1}

run["parameters"] = params
run["sys/name"] = "basic-colab-example"
run["sys/tags"].add(["colab", "intro"])

for epoch in range(100):
    run["train/loss"].log(0.99 ** epoch)

run["train/accuracy"] = 0.95
run["valid/accuracy"] = 0.93

# %%
tmp = os.listdir(".neptune/offline")[0]
cmd = f"neptune sync -p {project} --run offline/{tmp}"
subprocess.check_output(cmd, shell = True)

# %%
