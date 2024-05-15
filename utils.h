/// @author Ashish Kumar Meshram

#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

// ----------------------------- Memory function ---------------------------- //

/// @brief Allocate aligned memory
/// @param N[in] Number of elements
/// @param Nsize[in] Size of each element
/// @param alignment[in] aligment bytes
/// @return 
void* malloc_align(size_t N, size_t Nsize, size_t alignment);

// ---------------------------- Printing function --------------------------- //

/// @brief 
/// @param s 
/// @param dtype 
/// @param num 
/// @param x 
void printf_intrinsic(char *s, char *dtype, int num, void *x);

/// @brief 
/// @param[in] x 
/// @param[in] s 
/// @param[in] N1 
/// @param[in] N2 
void print_array(char *dtype, void *x, char *s, int N1, int N2);

/// @brief 
/// @param ptr 
/// @param dtype 
void read_binary(void *ptr, char *dtype);

// ------------------------- Random Number Generation ----------------------- //

/// @brief Generate Uniformly distributed random number between a and b
/// @param[in] a Lower limit [Float type]
/// @param[in] b Upper limit [Float type]
/// @param[out] c output array [Float type]
/// @param[in] N Number of elements in an array [Integer type]
void unif_randf(float a, float b, float *c, int N);

// ----------------------------- Number Conversion -------------------------- //

/// @brief Converts single precision floating point number to Q15 fixed point
/// number
/// @param[in] xfp floating point number 
/// @return Q15 fixed point number
int16_t fp_to_q15(float xfp);

/// @brief Converts Q15 fixed point number to single precision floating point
/// number
/// @param xFP 
/// @return 
float q15_to_fp(int16_t xFP);

/// @brief Convert floating point complex numbers to fixed-point
/// @param xfp 
/// @param xFP 
/// @param N 
/// @param maxval 
/// @param scval 
void float_to_fixed_point(float *xfp, int16_t *xFP, int N, float maxval, int scval);

/// @brief Convert fixed-point complex numbers to floating point
/// @param xFP 
/// @param xfp 
/// @param N 
/// @param maxval 
/// @param scval 
void fixed_point_to_float(int16_t *xFP, float *xfp, int N, float maxval, int scval);

/// @brief 
/// @param[in] x 
/// @param[in] y 
/// @param[in] N 
/// @return 
float calculateMSE(float *x, float *y, int N);

// ----------------------- SISD Complex Multiplication ---------------------- //
/// @brief Generate Uniform random number between a and b
/// @param[in] x Lower limit [Double type]
/// @param[in] y Upper limit [Double type]
/// @param[out] z Upper limit [Double type]
void complex_mult_float(float *x, float *y, float *z, int N);

/// @brief 
/// @param x 
/// @param y 
/// @param z 
/// @param N 
void complex_mult_fixed_point(int16_t *x, int16_t *y, int16_t *z, int N);

// ----------------------- SIMD Complex Multiplication ---------------------- //
/// Single precision floating point SIMD routines for complex number multiplications
/// @brief Perform SIMD complex number multiplications between two single precision 
///        floating point numbers
/// @param[in] x complex number x [Float type]
/// @param[in] y complex number y [Float type]
/// @param[out] z complex number z [Float type]
/// @param[in] N Number of complex IQ samples [Integer]
void complex_mult_float_SSE(float *x, float *y, float *z, int N);

/// @brief 
/// @param x 
/// @param y 
/// @param z 
/// @param N 
void complex_mult_float_AVX256(float *x, float *y, float *z, int N);

/// @brief 
/// @param x 
/// @param y 
/// @param z 
/// @param N 
void complex_mult_float_AVX512(float *x, float *y, float *z, int N);

/// Fixed point SIMD routines for complex number multiplications

/// @brief 
/// @param x 
/// @param y 
/// @param z 
/// @param N 
/// @param shift 
void complex_mult_fixed_point_SSE(int16_t *x, int16_t *y, int16_t *z, int N, int shift);

/// @brief 
/// @param x 
/// @param y 
/// @param z 
/// @param N 
/// @param shift 
void complex_mult_fixed_point_AVX256(int16_t *x, int16_t *y, int16_t *z, int N, int shift);

/// @brief 
/// @param x 
/// @param y 
/// @param z 
/// @param N 
/// @param shift 
void complex_mult_fixed_point_AVX512(int16_t *x, int16_t *y, int16_t *z, int N, int shift);


// ------------------------------- Write ------------------------------------ //

/// @brief Write complex IQ samples in MATLAB m-files
/// @param filename 
/// @param varname 
/// @param c 
/// @param N 
void write_iq_to_mfile(const char *filename, const char *varname, float *c, int N);

void write_to_mfile(const char *filename, const char *varname, float *c, int N);

/// @brief 
/// @param x 
/// @param N 
/// @param y 
/// @param M 
void extract_element_from_array(float *x, int N, float *y, int M);

#endif //UTILS_H