#!/bin/bash
  #SBATCH --job-name='lidpd.sh'
#SBATCH -o /home/lorenzo.spampinato/scratch/0_lid-pd/outfile_lid_pd_2
#SBATCH -e /home/lorenzo.spampinato/scratch/0_lid-pd/errfile_lid_pd_CTL
# SBATCH -N1
#SBATCH -p compute



#SBATCH --time 12-00:00:00

# Specify the following variables
ENV_NAME="LID_PD"
VOLUME="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Preprocessing_post_ICA_HP_0.5Hz"
#VOLUME="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Dataset_bin"
#VOLUME="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Dataset_set"
PWD_APP="/home/lorenzo.spampinato/scratch/0_lid-pd/"
IMAGE_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/singularity_img/luigi/iside-thor-imgs_lidpdv1.sif"

DATA_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Preprocessing_post_ICA_HP_0.5Hz"
#DATA_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Dataset_bin"
#DATA_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Dataset_set"
LABEL_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Annotations"
SAVE_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Results"
INFO_PATH="/home/lorenzo.spampinato/MEDITECH/BSP/data/lid_pd/Results"


ONLY_CLASS="DYS"
ONLY_PATIENT="PD044,PD045"

# Specify the python scripts to run
#PYTHON_SCRIPT="C:\Users\Lorenzo\PycharmProjects\lid-pd-sleep-v0"
PYTHON_SCRIPT="lidpd_main.py"
PYTHON_ARGS="--data_path $DATA_PATH --label_path $LABEL_PATH --save_path $SAVE_PATH --info_path $INFO_PATH --only_class ${ONLY_CLASS} --only_patient ${ONLY_PATIENT}"

srun singularity exec \
    -B ${VOLUME}:/Preprocessing_post_ICA_HP_0.5Hz,${PWD_APP}:/app,${SAVE_PATH}:/Results,${LABEL_PATH}:/Annotations \
    --pwd /app ${IMAGE_PATH} \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${ENV_NAME} && \
    python /app/${PYTHON_SCRIPT} --data_path /Preprocessing_post_ICA_HP_0.5Hz --label_path /Annotations --save_path /Results --info_path /Results --only_class ${ONLY_CLASS} --only_patient ${ONLY_PATIENT}"