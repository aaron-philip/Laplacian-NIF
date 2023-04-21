"""
Nomenclature:

Creates all the submit and training scripts for various model architectures that were trained
to restore the field with 6 sensors.

1st Number: Number of Units in ShapeNet Layers
2nd Number: Number of Layers in ShapeNet
3rd Number: Number of Units in ParameterNet Layers
4th Number: Number of Layers in ParameterNet
5th Number: Rank of Latent Space
"""


import re
import os

print(os.getcwd())
extra = "6sensor"
for nsu in range(50, 51, 1):
    for nsl in range(4,5,1):
        for npu in range(10, 60, 20):
            for npl in range(2,5):
                for rls in range(5,30,10):

                    os.system("echo '#!/bin/bash --login\n#SBATCH --partition FRIB-nodes\n#SBATCH --exclusive\n#SBATCH --nodes=1\n#SBATCH --time=10:00:00\n#SBATCH --gres=gpu:1\n#SBATCH --job-name {0}_{1}_{2}_{3}\
                        \n\nmodule load GCCcore/12.2.0\nmodule load CUDA/11.8.0\
                        \n\nsource ~/.bashrc\nconda deactivate\nconda activate tddft-emulation\
                        \nexport PYTHONPATH=/mnt/scratch/philipaa/tddft-emulation/nif\ngrep MemTotal /proc/meminfo\
                        \nfree -h\nulimit -s unlimited\n\necho 'Running: {5}-{0}_{1}_{2}_{3}_{4}'\npython train_{5}-{0}_{1}_{2}_{3}_{4}.py' > submit_{5}-{0}_{1}_{2}_{3}_{4}.sb".format(nsu, nsl, npu, npl, rls, extra))
        
                    os.system("cp template6sensor_train.py train_{5}-{0}_{1}_{2}_{3}_{4}.py".format(nsu, nsl, npu, npl, rls, extra))
                    with open('template6sensor_train.py', 'r') as f:
                        txt = f.read()
                        txt = re.sub("N_P_Layers = 0", "N_P_Layers = %i" % npl, txt, 1)
                        txt = re.sub("N_P_Units = 0", "N_P_Units = %i" % npu, txt, 1)
                        txt = re.sub("N_S_Layers = 0", "N_S_Layers = %i" % nsl, txt, 1)
                        txt = re.sub("N_S_Units = 0", "N_S_Units = %i" % nsu, txt, 1)
                        txt = re.sub("Rank_Linear = 0", "Rank_Linear = %i" % rls, txt, 1)

                    with open('train_{5}-{0}_{1}_{2}_{3}_{4}.py'.format(nsu, nsl, npu, npl, rls, extra), 'w') as f:
                        f.write(txt)
