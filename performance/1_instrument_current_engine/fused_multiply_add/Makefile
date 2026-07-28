CC     := clang
CFLAGS := -O2 -mavx2 -mfma

.PHONY: all clean

all: peak_fma

peak_fma: driver.c fma_kernel.s
	$(CC) $(CFLAGS) driver.c fma_kernel.s -o peak_fma

clean:
	rm -f peak_fma
