# MultiTaskRL

## Local Installation
To install with conda:
```commandline
pip install -r requirements.txt
pip install -e custom_envs
```
To install with Docker:
1. Install Docker
2. In `docker_build_local.sh` and `docker_run_local.sh`, replace `nicholascorrado` with your Docker Hub username.
3. Run `./docker_build.sh`
4. Run `./docker_run.sh`

## CHTC Usage

Build your Docker image:
1. Install Docker
2. In `docker_build_chtc.sh`, replace `nicholascorrado` with your Docker Hub username.
3. Run `./docker_build_chtc.sh`

After pushing your Docker image:
1. In `job.sub`, update `container_image = docker://nicholascorrado/multitask-rl` to point to your new Docker image.
2. Generate a text file containing the commands you want to run and place it in `chtc/commands`. 
You can use a script to generate many commands. See `chtc/gen_commands` for an example.
The file should be formatted as follows: 
```commandline
MEMORY,DISK,PYTHON COMMAND
```
```commandline
1,10,python ppo_continuous_action.py --env_id Hopper-v5 --learning_rate 0.0003 --num_steps 8192 --output_subdir lr_0.0003/ns_8192 --total_timesteps 1000000
```
3. Login to chtc, and `git clone` this repo to your submit node.
4. `cd` to the `chtc` directory, and run `submit.sh {commands_file.txt} {n}` where `commands_file.txt`
is in the `commands` directory, and `n` denote the number of trials you want to run each command for. For instance,
`n=10` will run each experiment 10 times with seeds 0-9.
5. Results will be saved to `results/{commands_file}` and job logs are saved to `results/{commands_file}/logs`.

After a job finishes, check the memory and disk usage in the the `.log` output to ensure you requested an appropriate amount.
If you requested much more than was actually used by the job, reduce it.

## CHTC Interactive Usage
1. In `job_i.sub`, update `container_image = docker://nicholascorrado/multitask-rl` to point to your new Docker image.
2. Run `./submit_interactive.sh` to start an interactive session.
