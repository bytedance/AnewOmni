#/bin/bash
########## parsing directory of the script ##########
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR

########## create environment ##########
echo "Creating environment"
cd $SCRIPT_DIR
conda create -p ./env python=3.11 --yes
source activate base
conda activate ./env
pip install protenix

########## download checkpoints ##########
echo "Try running the demo case for preparation"
PROTENIX_ROOT_DIR=$SCRIPT_DIR/params protenix pred \
    -i $SCRIPT_DIR/demo.json \
    -o $SCRIPT_DIR/test_out \
    -n protenix_mini_esm_v0.5.0 \
    --use_msa false \
    --use_seeds_in_json true

######### clean up ##########
rm -r $SCRIPT_DIR/esm_embeddings
