# Instruction for using csc taito
Logging into taito
1. CPU only
```bash
ssh uname@taito.csc.fi
```
1. GPU
```bash
ssh uname@taito.csc.fi
```

## Submitting a bash job

1. Setting up python env
    * List python env
        ```bash
        module spider python
        ```
    * For taito
        ```bash
        module load python-env/version
        ```
    * [Python with ML library](https://research.csc.fi/-/mlpython)
        ```bash
        module purge
        module load python-env/3.6.3-ml
        ```

## Bash Job script
```Bash
#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
# Estimate of how long your GPU job will take to execute. 30 min is
# probably fine
#SBATCH -t 00:30:00
#SBATCH --mem=10000
#SBATCH -J pos_tagger_vn
#SBATCH -o pos_tagger_vn.out.%j
#SBATCH -e pos_tagger_vn.err.%j
#SBATCH --gres=gpu:k80:1
#SBATCH                                                                                                                                   

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9
cd POSTagger_Pytorch
pip3 install pipenv
pipenv install
python3 main.py --language Vietnamese --gpu True --save_model True
```

```Bash
#!/bin/bash
module purge
module load cuda-env
module load python-env/3.6.3-ml
pip install conllu --user
pip install xmltodict --user
cd POSTagger_Pytorch
chmod +x ./downloadData.sh
./downloadData.sh
python main.py --language all --gpu True --save_model True
```
