CC     = gcc
TARGET = adaboost_cpu
CFLAGS = -g -Wall -Werror -pedantic-errors

all:
	$(CC) imagelib_cpu.c adaboost_cpu.c -lm -o $(TARGET)
clean:
	rm -f $(TARGET) $(TARGET).exe .*~
