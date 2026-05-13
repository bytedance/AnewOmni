#/bin/bash
########## parsing directory of the script ##########
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

########## create environment ##########
echo "Creating environment"
cd $SCRIPT_DIR
conda create -p ./env python=3.10 --yes
source activate base
conda activate ./env
pip install boltz[cuda]==0.4.1 torch==2.5.1 gemmi -U --no-cache-dir

######### download weights ##########
python download_ckpts.py