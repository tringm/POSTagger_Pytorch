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
module purge
module load python-env/3.6.3-ml
pip install conllu --user
pip install xmltodict --user
cd POSTagger_Pytorch
chmod +x ./downloadData.sh
./downloadData.sh
python main.py --language all --gpu True
```
