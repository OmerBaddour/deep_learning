CC       := clang
CFLAGS   := -Wall -Wextra
AVXFLAGS := -O3 -mavx2 -mfma

ALL      := normal compiler_auto_vectorization intrinsics
DUMPS    := $(addsuffix .txt,$(ALL))

.PHONY: all objdump clean

all: $(ALL)

normal: float_array_sum.c
	$(CC) $(CFLAGS) -O0 float_array_sum.c -o $@

compiler_auto_vectorization: float_array_sum.c
	$(CC) $(CFLAGS) $(AVXFLAGS) float_array_sum.c -o $@

intrinsics: float_array_sum_intrinsics.c
	$(CC) $(CFLAGS) $(AVXFLAGS) float_array_sum_intrinsics.c -o $@

# Pattern rule: each dump depends on its binary, so make builds whatever is
# missing and skips whatever is already up to date.
%.txt: %
	objdump -d $< > $@

objdump: $(DUMPS)

clean:
	rm -f $(ALL) $(DUMPS)
