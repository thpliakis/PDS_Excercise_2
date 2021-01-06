SOURCE = V1_asynchronous
NPROCS = 3

all:
	make run

$(SOURCE).out:	$(SOURCE).c
	mpicc -O3 $< -o $(SOURCE).out -lblas -lm

run:	$(SOURCE).out
#	mpiexec -machinefile mpd.hosts -n $(NPROCS) ./$(SOURCE).out
	mpiexec  -n $(NPROCS) ./$(SOURCE).out -quiet

clean:
	rm -f *~ *.out


