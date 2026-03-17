import runpy


if __name__ == "__main__":
    print("Please use one dedicated script per ablation:")
    print("  - chtc/gen_commands/gridworld_ablate_dro_eta.py")
    print("  - chtc/gen_commands/gridworld_ablate_dro_step_size.py")
    print("  - chtc/gen_commands/gridworld_ablate_dro_num_steps.py")
    print("  - chtc/gen_commands/gridworld_ablate_num_steps.py")
    print("")
    print("Running default: gridworld_ablate_dro_eta.py")
    runpy.run_path("gridworld_ablate_dro_eta.py", run_name="__main__")

