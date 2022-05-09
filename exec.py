#%%
from utils import  get_hyper_params
import subprocess


if __name__ == "__main__":

    hyper_params_dict = get_hyper_params()
    with open("run_frcnn_slurm.sh", "r") as rsh:
        run_frcnn_slurm = rsh.readlines()

    experiment_settings = [j.split("_") for j in list(hyper_params_dict.keys())]
    for experiment_setting in experiment_settings:
        base_model = experiment_setting[0]
        dataset_name = experiment_setting[1]
        bash_dataset_name = dataset_name.replace("/", "")
        run_frcnn_slurm[len(run_frcnn_slurm)-1] = f"python /home1/wonhyung64/Github/Faster_R-CNN/main.py --base-model {base_model} --dataset-name {dataset_name}\n"

        with open(f"run_frcnn_slurm_{base_model}_{bash_dataset_name}.sh", "w") as wsh:
            wsh.writelines(run_frcnn_slurm)
        cmd = f"sbatch --job-name=frcnn --partition=hgx --gres=gpu:hgx:1 run_frcnn_slurm_{base_model}_{bash_dataset_name}.sh"
        result = subprocess.check_output(cmd, shell = True)
        print(cmd)