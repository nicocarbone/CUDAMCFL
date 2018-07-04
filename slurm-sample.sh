#!/bin/bash

# Ejecuta un programa con uso de gpu, usando una gpu por nodo, en 4 nodos.

# Correr con `sbatch /ruta/a/este/script

#SBATCH --job-name="CUDAMCFL"
#SBATCH --nodes=1
#SBATCH --workdir=/home/ncarbone/CUDAMCFL
#SBATCH --error="cudamcfl-gpu-%j.err"
#SBATCH --output="cudamcfl-gpu-%j.out"
#SBATCH --partition=fast-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=60:00

echo "trabajo \"${SLURM_JOB_NAME}\""
echo "    id: ${SLURM_JOB_ID}"
echo "    partición: ${SLURM_JOB_PARTITION}"
echo "    nodos: ${SLURM_JOB_NODELIST}"
echo
date +"inicio %F - %T"

echo "
--------------------------------------------------------------------------------
"

# INICIO VARIABLES IMPORTANTES
#
# NO TOCAR. No se aceptan reclamos en caso de modificar estas líneas. Deberán
# incluirlas siempre, hasta próximo aviso.
#
[ -r /etc/profile.d/odin-users.sh ] && . /etc/profile.d/odin-users.sh
#
# FIN VARIABLES IMPORTANTES

# El path al programa es `/home/USUARIO/ejemplos/programa-gpu`. Como más arriba
# seteamos el directorio de trabajo a `/home/USUARIO/ejemplos`, el programa se
# encuentra en el directorio de trabajo, por lo que basta poner
# `./programa-gpu`.
srun ./cuda_fl_k20 sample2.mci
echo "
--------------------------------------------------------------------------------
"

date +"fin %F - %T"
