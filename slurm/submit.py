import datetime
import subprocess
import time


def follow(thefile):
    thefile.seek(0, 2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line


# submits the job to the slurm queue
if __name__ == "__main__":

    # job_name = sys.argv[1]
    # script_name = sys.argv[2]
    # num_cores = sys.argv[3]
    # num_nodes = sys.argv[4]
    # time_limit = sys.argv[5]
    # memory_limit = sys.argv[6]
    # partition = sys.argv[7]
    output_name = (
        "/work/rleap1/jakob.krude/projects/remote/rgnet/slurm/run"
        + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    ) + ".out"

    job_file_text = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --output={output_name}
source /work/rleap1/jakob.krude/projects/rgnet/venv/bin/activate
export PYTHONPATH=/work/rleap1/jakob.krude/projects/remote/rgnet/src/:$PYTHONPATH
cd /work/rleap1/jakob.krude/projects/remote/rgnet/
python /work/rleap1/jakob.krude/projects/remote/rgnet/examples/run_supervised.py"""
    # create the script
    with open(f"job", "w") as f:
        f.write(job_file_text)
    subprocess.run(["sbatch", "job"])
    print("Output can be found at " + output_name)
