#ifndef ASSIGNMENTS_CUH
#define ASSIGNMENTS_CUH

#include <cuda_runtime.h>
#include <inttypes.h>
#define ASSIGNMENT_COUNT(a) ((int) ceil((a)/(LITS_PER_WORD*1.0)))
#define LITS_PER_WORD (16)

typedef enum { LIT_UNSET = 0, LIT_TRUE = 1, LIT_FALSE = 2, LIT_ERROR = 3 } litval_t;
typedef uint32_t assignment_t;

__device__ __host__  litval_t get_lit(const assignment_t* a, int32_t lit);
__device__ __host__ void set_lit(assignment_t* a, int32_t lit, litval_t value);
__device__ __host__ int is_lit_sated(const assignment_t* const  a, const int32_t lit);
__device__ __host__ int is_lit_sated_complete(const assignment_t* const a, const int32_t lit, const uint64_t altBWAssignment);
__device__ __host__ int clear_assignment(assignment_t* a, uint32_t litCount);
#endif