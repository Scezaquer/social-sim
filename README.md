# social-sim

All the simulator's code is contained in the src folder

Setup (NOTE: if you do not plan on running this yourself you may skip this.
These are some fairly large libraries so don't install them unless you HAVE to.
You also need cuda, I know it works for at least CUDA 8.0 and CUDA 12.0).

```
python -m venv ENV-simple-sim
source ENV-simple-sim/bin/activate
python -m pip install -r requirements.txt
```

Running a simulation with default arguments (you'll need a gpu capable of running an 8B model)

```
python src/main.py
```

For more complicated possible argument setups there are examples in the `bash_job_scripts` folder.