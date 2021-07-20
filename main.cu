#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "assignments.cuh"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>



#ifdef __INTELLISENSE__
//THIS IS JUST A WORKARROUND. INTELISENSE CLAIMS __syncthreads() IS NOT DEFINED BUT IT COMPILES FINE (IT IS). 
//WE JUST DEFINE A MACRO TO MAKE THE ERROR ON THE EDITOR GO AWAY
#define __syncthreads() printf("you shouldn't be seeing this");exit(1);
#define __syncwarp() printf("you shouldn't be seeing this");exit(1);
#endif

#define THREAD_INDX_1D_GRID_2D_BLOCK (blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x)
#define THREAD_INDX_2D_BLOCK (threadIdx.y * blockDim.x + threadIdx.x)

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif
#define SIGN(a)((a)<0?-1:1)
#define CHECK_CUDA_ERROR(expr) \
{ \
	cudaError_t a__cuda_error = expr; \
	if (a__cuda_error != cudaSuccess) { \
		fprintf(stderr,"ERROR on line %d file %s: [%s]\n", __LINE__, __FILE__, cudaGetErrorString(a__cuda_error)); \
		exit(a__cuda_error); \
	} \
}

#define BLOCK_DIM 32
#define BLOCK_CNT 136


typedef struct {
	int32_t lits[3];
} clause3_t;
typedef struct {
	uint32_t litCount, length;
	clause3_t* clauses;
} expression_t;

enum sat_result { SAT_SAT, SAT_UNSAT, SAT_ERROR, SAT_UNDETERMINED };

int readExpressionFromFile(FILE*, expression_t**);
void print_expression(expression_t*);
int remap_unsat_lits(expression_t*, assignment_t*);
void swap(expression_t* e, assignment_t* a, int32_t j, int32_t i);
void solver_driver(expression_t*, assignment_t*, uint8_t);
//Expression, partialAssignment, unsetVarCount (Assumes remaped), number_of_threads_started 
__global__ void bf_solve_kernel(const expression_t* const, assignment_t*, const uint8_t, const uint32_t);
__device__  __inline__ int d_check(const expression_t* const, assignment_t*, const uint64_t);
void test();


int assgn_dist_score(int32_t one, int32_t two) {
	one = abs(one);
	two = abs(two);
	const int32_t assgnWordIndxOne = ((one - 1) / LITS_PER_WORD);
	const int32_t assgnWordIndxTwo = ((two - 1) / LITS_PER_WORD);
	const uint32_t dist = abs(assgnWordIndxOne - assgnWordIndxTwo);
	if (dist <= 5) return 5-dist;//4-dist means the closer to 4 distance the worse the returned score. dist 5 = score 0.
	if (dist == 0) return 10;//Very good, the two are close together.
	return 5-dist;//The further appart (out of the 4 distance box) they are, the worse the score. (negative)

}
//Metric: distance of literals in assignment. An assignment word contains assignments to N literals.
//If the lits of a clause are within the same assignment word that's good.
//If they're within 4 words that's okay
//If they're off more than that, then it's bad. 
int memory_distance_comparator(const void* a, const void* b) {
	clause3_t* c1 = (clause3_t*)a;
	clause3_t* c2 = (clause3_t*)b;
	int c1d = assgn_dist_score(c1->lits[0], c1->lits[1]) + assgn_dist_score(c1->lits[1], c1->lits[2]) + assgn_dist_score(c1->lits[0], c1->lits[2]);
	int c2d = assgn_dist_score(c2->lits[0], c2->lits[1]) + assgn_dist_score(c2->lits[1], c2->lits[2]) + assgn_dist_score(c2->lits[0], c2->lits[2]);
	return (c1d - c2d);
}
void sort_clauses(expression_t* e) {
	qsort(e->clauses, e->length, sizeof(clause3_t), memory_distance_comparator);
}


int main(int argc, char* argv[]) {
	expression_t* e;
	assignment_t* a;

	int err = readExpressionFromFile(fopen("R:/unsat30-400.cnf", "r"), &e);
	if (err) {
		printf("ERROR reading file: %d\n", err);
		return err;
	}
	a = (assignment_t*) malloc(sizeof(assignment_t) * ASSIGNMENT_COUNT(e->litCount));

	sort_clauses(e);

	for (int i = 0; i < e->length; i++) {
		clause3_t* c1 = &e->clauses[i];
		int c1d = assgn_dist_score(c1->lits[0], c1->lits[1]) + assgn_dist_score(c1->lits[1], c1->lits[2]) + assgn_dist_score(c1->lits[0], c1->lits[2]);
		printf("(%3d,%3d,%3d) -> %d,\n", c1->lits[0], c1->lits[1], c1->lits[2], c1d);
	}

	a = (assignment_t*)malloc(sizeof(assignment_t) * ASSIGNMENT_COUNT(e->litCount));
	clear_assignment(a, e->litCount);

	if (!a) {
		printf("Cannot allocate memory for assignment!\n");
		return -1;
	}

	
	int resp = remap_unsat_lits(e, a);
	if (resp <= 0 || resp > 64) {
		printf("Too many unsat lits found! Remaping failed; expression corrupted.");
		return EXIT_FAILURE;
	}
	printf("Remapped literals: %u\n", resp);
	solver_driver(e, a, resp);
	return EXIT_SUCCESS;
}



void swap(expression_t* e, assignment_t* a, int32_t j, int32_t i) {
	if (j == i)
		return;
	for (uint32_t o = 0; o < e->length; o++) {
		int32_t* cl = e->clauses[o].lits;

		for (uint32_t x = 0; x < 3; x++) {
			if (cl[x] == j || cl[x] == -j)
				cl[x] = SIGN(cl[x]) * i;
			else if (cl[x] == i || cl[x] == -i)
				cl[x] = SIGN(cl[x]) * j;
		}
	}
	litval_t s = get_lit(a, j);
	set_lit(a, j, get_lit(a, i));
	set_lit(a, i, s);
}


//TODO: make it a kernel where each thread has a clause, and it checks if any of the lits is in the remapping
//and replaces them accordingly. max 64 kernel launches can run in parallel and synchrn should be called before returning.
int remap_unsat_lits(expression_t* e, assignment_t* a) {
		uint32_t i, j;
	for (i = 0, j = 0; j <= e->litCount && i <= 64; j++) {
		if (j > i && get_lit(a, j) == LIT_UNSET) {
			i++;
			swap(e, a, j, i);
			j = 0;//restart search
		}
	}
	assert(i <= 64); 
	return i;

}

void print_expression(expression_t* e) {
	printf("p cnf %d %d\n", e->litCount, e->length);
	for (unsigned i = 0; i < e->length; i++)
		printf("%d %d %d\n", e->clauses[i].lits[0], e->clauses[i].lits[1], e->clauses[i].lits[2]);
	printf("\n\n");
}
__global__ void printExp(const expression_t* const g_e) {
	printf("Expression of length %u, with %u lits:\n", g_e->length, g_e->litCount);
	for (uint32_t i = 0; i < g_e->length; i++) {
		printf("(%d,%d,%d)\n", g_e->clauses[i].lits[0], g_e->clauses[i].lits[1], g_e->clauses[i].lits[2]);
	}
}
void solver_driver(expression_t* e, assignment_t* a, uint8_t unsetCnt) {

	cudaSetDevice(1);
	CHECK_CUDA_ERROR(cudaFuncSetAttribute(bf_solve_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
	CHECK_CUDA_ERROR(cudaFuncSetCacheConfig(bf_solve_kernel, cudaFuncCachePreferL1));
	expression_t* d_e;
	clause3_t* d_e_c;
	assignment_t* d_a;
	CHECK_CUDA_ERROR(cudaMalloc(&d_e, sizeof(expression_t)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_e_c, sizeof(clause3_t) * e->length));
	CHECK_CUDA_ERROR(cudaMalloc(&d_a, sizeof(assignment_t) * ASSIGNMENT_COUNT(e->litCount)));

	CHECK_CUDA_ERROR(cudaMemcpy(d_e, e, sizeof(expression_t), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_e_c, e->clauses, sizeof(clause3_t) * e->length, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(&d_e->clauses, &d_e_c, sizeof(clause3_t*), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, sizeof(assignment_t) * ASSIGNMENT_COUNT(e->litCount), cudaMemcpyHostToDevice));

	const uint32_t solverCount = BLOCK_DIM * BLOCK_DIM * BLOCK_CNT;
	const dim3 block(BLOCK_DIM, BLOCK_DIM);

	printf("Limit: %u", (uint32_t)ceil((1LLU << unsetCnt) / (double)solverCount));
	
#ifdef BENCHMARK_ROUNDS
	for (int i = 0; i < BENCHMARK_ROUNDS; i++) {
#endif
		bf_solve_kernel CUDA_KERNEL(BLOCK_CNT, block)(d_e, d_a, unsetCnt, solverCount);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef BENCHMARK_ROUNDS
	}
#endif
	CHECK_CUDA_ERROR(cudaFree(d_e));
	CHECK_CUDA_ERROR(cudaFree(d_e_c));
	CHECK_CUDA_ERROR(cudaFree(d_a));
}



/*
To try:
1. Start N threads for some constant N each computing their own assignment, and checking the clause:
	- Check if synchronizing the warp before checking next assignment helps with parallelism at all (i.e. make everyone wait so that they can all spend the most time doing the same op)
2. Start X threads s.t. X = C*f (where f is some constant >= 1) and on each step a new assignment is created and each thread checks it's clause.
	- Who generates the assignment is TBA. We don't wana start a new thread.
	- Saves a thread from going over a loop but not much else.
*/
__global__ void bf_solve_kernel(const expression_t* const g_e, assignment_t* g_a, const uint8_t unsetCnt, const uint32_t solvers) {
	const uint64_t limit = (uint32_t)ceil((1LLU << unsetCnt) / (double)solvers); //Isn't it better to ((1LLU<<unsetcnt)/solvers)+!!(unsetcnt%solvers)
	uint64_t i, local_assignmen = THREAD_INDX_1D_GRID_2D_BLOCK;
	for (i = 0; i < limit; i++) {
		if (local_assignmen > (1LLU << unsetCnt)) break;
		if (d_check(g_e, g_a, local_assignmen)) {
			printf("SAT!: %lux\n", local_assignmen);
			break;
		}
		local_assignmen += solvers;
		__syncwarp();
	}
}

__device__ __forceinline__ int d_check(const expression_t* const g_e, assignment_t* const g_a, const uint64_t assgn) {
	bool whole = 1, curr = 0;
	uint32_t i = 0;
	for (; whole & (i < g_e->length); i++, curr = 0) {
		int32_t a = g_e->clauses[i].lits[0];
		curr |= (is_lit_sated_complete(g_a, a, assgn));

		a = g_e->clauses[i].lits[1];
		curr |= (is_lit_sated_complete(g_a, a, assgn));

		a = g_e->clauses[i].lits[2];
		curr |= (is_lit_sated_complete(g_a, a, assgn));

		whole &= curr;
	}
	return whole;
}


int getLines(char*** arr, unsigned long long* r_lineCount, FILE* f) {
	if (!f || feof(f) || ferror(f))
		return -1;
	fseek(f, 0L, SEEK_END);
	size_t fileSize = ((size_t)ftell(f)) + 1L;//+1, ftel is 0 indexed.
	rewind(f);

	char* in = (char*)malloc(sizeof(char) * fileSize + 3);
	memset(in, 0, sizeof(char) * fileSize + 3);
	if (!in)
		return -2;

	fread(in, sizeof(char), fileSize, f);
	in[fileSize - 1] = '\n';//HACK. Just so the while loop doesn't finish one-off we introduce a fake line (anything after the last \n is ignored basically.)
	in[fileSize] = '\0';


	unsigned long long lineCount = 0;//should be 1 but we add an artificial \n at the end of the file so all good.
	for (unsigned long long i = 0; i < fileSize; i++)
		if (in[i] == '\n')
			lineCount++;
	*r_lineCount = lineCount;

	char** l_arr = (char**)malloc(sizeof(char*) * lineCount);
	if (!l_arr)
		return -2;
	unsigned long long lines = 0;
	char* strt = in;
	char* nl = NULL;
	while ((nl = strchr(strt, '\n'))) {
		//nl will always be strt+diff
		size_t diff = (nl - strt) / sizeof(char);
		if (diff == 0) {
			l_arr[lines] = NULL;
		} else {
			l_arr[lines] = (char*)malloc(sizeof(char) * (diff + 1));
			if (!l_arr[lines])
				return -2;

			memcpy(l_arr[lines], strt, diff);
			l_arr[lines][diff] = '\0';
		}
		++lines;
		strt = nl + 1;
	}
	*arr = l_arr;
	free(in);
	return 0;
}

//Assumes complete 3-sat
int readExpressionFromFile(FILE* f, expression_t** putWhere) {
	unsigned long long lineCount;
	char** lines_arr = NULL;
	unsigned long vars = 0, clausec = 0;
	expression_t* e = NULL;

	int err = getLines(&lines_arr, &lineCount, f);
	if (err) goto done;

	e = (expression_t*)malloc(sizeof(expression_t));
	for (uint32_t cl = 0, li = 0; li < lineCount; li++) {
		char* line = lines_arr[li];
		if (!line)
			continue; //Skip blank line
		//skip spaces, tabs at start 
		for (; line[0] == ' ' || line[0] == '\t'; line++);
		//skip comment
		if (line[0] == 'c')
			continue;
		//End of file???
		else if (line[0] == '%')
			break;
		else if (line[0] == 'p') {
			if (vars == 0 && clausec == 0) {
				int match = sscanf(line, "p cnf %lu %lu", &vars, &clausec);
				if (match != 2) { err = -4;	goto done; }
				e->clauses = (clause3_t*)malloc(sizeof(clause3_t) * clausec);
				e->length = clausec;
				e->litCount = vars;
			} else { err = -5; goto done; } //duplicate problem line 
		} else {
			if (vars == 0 || clausec == 0) {
				err = -6; goto done;
			} else {
				int32_t a, b, c;
				int match = sscanf(line, "%d %d %d 0", &a, &b, &c);
				if (match != 3) {
#ifndef NDDEBUG 
					printf("Problem with line %u\n", li);
#endif
					err = -7;
					goto done;
				}
				if (cl >= clausec) { err = -8; goto done; }
				if (a == 0 || b == 0 || c == 0) { err = -9; goto done; }
				e->clauses[cl].lits[0] = a;
				e->clauses[cl].lits[1] = b;
				e->clauses[cl].lits[2] = c;
				cl++;
			}
		}
	}
done:

	if (lines_arr) {
		for (int i = 0; i < lineCount; i++) {
			if (lines_arr[i])
				free(lines_arr[i]);
		}
		free(lines_arr);
	}
	if (err) {
		if (e) {
			if (e->clauses) {
				free(e->clauses);
			}
			free(e);
		}
	} else {
		*putWhere = e;
	}
	return err;
}

void testIsLitSated() {
	assignment_t a[2];
	//clear_assignment(a, 15);
	set_lit(a, 1, LIT_UNSET);
	if ((is_lit_sated(a, 1)) == (0)) printf("Passed.\n"); else printf("Failed.\n");
	set_lit(a, 1, LIT_TRUE);
	if ((is_lit_sated(a, 1)) == (1)) printf("Passed.\n"); else printf("Failed.\n");
	set_lit(a, 1, LIT_FALSE);
	if ((is_lit_sated(a, 1)) == (0)) printf("Passed.\n"); else printf("Failed.\n");
	set_lit(a, 1, LIT_ERROR);
	if ((is_lit_sated(a, 1)) == (0)) printf("Passed.\n"); else printf("Failed.\n");

	set_lit(a, 1, LIT_UNSET);
	if ((is_lit_sated(a, -1)) == (0)) printf("Passed.\n"); else printf("Failed.\n");
	set_lit(a, 1, LIT_TRUE);
	if ((is_lit_sated(a, -1)) == (0)) printf("Passed.\n"); else printf("Failed.\n");
	set_lit(a, 1, LIT_FALSE);
	if ((is_lit_sated(a, -1)) == (1)) printf("Passed.\n"); else printf("Failed.\n");
	set_lit(a, 1, LIT_ERROR);
	if ((is_lit_sated(a, -1)) == (0)) printf("Passed.\n"); else printf("Failed.\n");
}

void testFastABSonHost() {
#define FAST_32_BIT_ABS(a)((((a)>>31)^a)-((a)>>31))

	for (int i = 0; i < pow(2, 31); i++) {
		if (abs(-i) != FAST_32_BIT_ABS(-i) || abs(i) != FAST_32_BIT_ABS(i)) {
			printf("CPUFailed for %d\n", i);
		}
	}
	printf("CPUPassed");
}
__global__ void testFastABSonDevice() {
#define FAST_32_BIT_ABS(a)((((a)>>31)^a)-((a)>>31))
	for (int i = 0; i < 2147483648; i++) {
		if (abs(-i) != FAST_32_BIT_ABS(-i) || abs(i) != FAST_32_BIT_ABS(i)) {
			printf("GPUFailed for %d\n", i);
		}
	}
	printf("GPUPassed");
}
