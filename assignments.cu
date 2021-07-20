#include "assignments.cuh"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

//The two different definitions seem to have no noticable performance differnce.
//abs(a) has a tiiiny performance drop vs the other, but it's not worth being so hw-implementation specific 
//for it.
//#define FAST_32_BIT_ABS(a)((((a)>>31)^a)-((a)>>31))
#define FAST_32_BIT_ABS(a) abs(a)

//Exploits the fact LIT_TRUE is 01 and LIT_FALSE is 10. 
//
//Right side: ((l & 1) ^ (lit < 0))
//The LSB of LIT_TRUE and LIT_FALSE are true and false respectively.
//lit<0 is 1 if lit is negated, 0 otherwise, l XOR'd with (lit<0) will yield
//0^1 when not negated and true, and 1^0 when negated and false.
//
//Left side: ((l>>1) ^ (l & 1))
//To capture the other cases (LIT_ERROR=11, LIT_UNSET=00) and return 0 in both, 
//we get the two bits of error and unset as separate bits, and XOR them together
//resulting in 1 only if LIT_TRUE or LIT_FALSE.
//
//Note: On compute capability 7.X to 8.6 (currently), we get 64 "results per clock cycle per multiprocessor" on
//AND OR and XOR: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions__throughput-native-arithmetic-instructions
__device__ __host__ int is_lit_sated(const assignment_t* const a, const  int32_t lit) {
	int l = get_lit(a, lit);
	return ((l >> 1) ^ (l & 1)) & ((l & 1) ^ (lit < 0));
}

__device__ __host__ int is_lit_sated_complete(const assignment_t* const a, const int32_t lit, const uint64_t altBWAssignment) {
	int l = get_lit(a, lit);
	//TODO: lit<0 appears to be branching (according to visual profiler on the 1080's) so
	//instead, try (i>>31)&0x1 to get the top bit. If it's 1 it's negative, otherwise positive. Relies on how
	//the numbers are repsesented!
	return ((l == LIT_UNSET) & (((lit < 0) & (~((altBWAssignment >> ((-lit) - 1)) & 1))) | ((lit > 0) & (((altBWAssignment >> (lit - 1)) & 1))))) | ((l != LIT_UNSET) & ((l >> 1) ^ (l & 1)) & ((l & 1) ^ (lit < 0)));
}


__device__ __host__ int clear_assignment(assignment_t* a, uint32_t litCount) {
	if (!a)
		return 0;
	//Abuse the fact LIT_UNSET is 0 and zero out memory. If LIT_UNSET changes, then: 
	//uint32_t mask = LIT_UNSET | (LIT_UNSET<<2) | (LIT_UNSET<<4) | (LIT_UNSET<<6) | (LIT_UNSET<<8) | (LIT_UNSET<<10) | (LIT_UNSET<<10) |  ...
	memset(a, 0, sizeof(assignment_t)* ASSIGNMENT_COUNT(litCount));
	return 1;
}

__device__ __host__ litval_t get_lit(const assignment_t* a, int32_t lit) {
	lit = FAST_32_BIT_ABS(lit);
	const uint32_t word = ((lit - 1) / LITS_PER_WORD);
	const uint32_t indx = (lit - 1) % LITS_PER_WORD;
	return (litval_t)((a[word] >> (indx * 2)) & 3);
}

__device__ __host__ void set_lit(assignment_t* a, int32_t lit, litval_t value) {
	lit = FAST_32_BIT_ABS(lit);
	const uint32_t word = ((lit - 1) / LITS_PER_WORD);
	const uint32_t shifts = 2 * ((lit - 1) % LITS_PER_WORD);
	a[word] = (a[word] & ~(3 << shifts)) | (value << shifts);
}
