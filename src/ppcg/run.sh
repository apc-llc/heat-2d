# 1) Copy heat-2d.c to  in.c 
# 2) open in.c and write  "#define T  replace"  where T is defined
# 3) replace in ../common.mk  "-msse3" with "-march=native"

#for i in {2..40} 
#rm -rf out_gpu
#for i in {1..40}
for i in $(seq 1000 100 10000)
do
export OMP_NUM_THREADS=1
echo $i | ./heat-2d >> out_gpu
done

# to get the results without vectorization
# change in ../common.m the OPT_FLAGS := -fno-tree-vectorize
