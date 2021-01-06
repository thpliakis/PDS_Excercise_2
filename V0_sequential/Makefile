CC = gcc
SOURCE = V0_sequential
CFLAGS =  -lblas -lm

all:  $(SOURCE).out
	./$(SOURCE).out

$(SOURCE).out:	$(SOURCE).c
	$(CC) -o3  $< -o $(SOURCE).out $(CFLAGS)


clean:
	rm -f *~ *.out


