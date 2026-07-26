CC     := clang
ALL    := normal compiler_auto_vectorization intrinsics

.PHONY: all normal compiler_auto_vectorization intrinsics objdump clean

all: $(ALL)

normal: float_array_sum.c
	$(CC) float_array_sum.c -o normal

compiler_auto_vectorization: float_array_sum.c
	$(CC) -O3 -mavx2 -mfma float_array_sum.c -o compiler_auto_vectorization

# TODO: figure out how to only execute on existing `target`s
objdump: $(ALL)
	@for target in $(ALL); do \
		objdump -d $$target > $$target.txt; \
	done

clean:
	rm -f $(ALL)
	@for target in $(ALL); do \
		rm -rf $$target.txt; \
	done
