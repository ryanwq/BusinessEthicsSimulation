# NK Ethics Simulation
Simulation code for "From Business-Ethics Tradeoffs to Simultaneous Improvement:
Ethical Beliefs, Innovation, and Organizational Learning."
## Requirements
Python 3.9 or later, plus three packages:
    pip install numpy matplotlib pandas
## Running the simulation
Run all commands from the `ethiraj_levinthal/` root folder.
Pilot run (30 replications, fast):
    python run_study.py --stage robustness --rb-variant A
Full replication (120 replications):
    python run_study.py --stage robustness --rb-variant A --reps 120
## Full instructions
See `Simulation Instructions.docx` for complete documentation of all stages,
variants, parameters, and expected outputs.
