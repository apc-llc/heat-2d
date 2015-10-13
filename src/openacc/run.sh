for i in $(seq 0 20 1000)
do
export OMP_NUM_THREADS=1
echo $i | ./heat-2d >> out_gpu
done

for i in $(seq 1100 100 10000)
do
export OMP_NUM_THREADS=1
echo $i | ./heat-2d >> out_gpu
done
