/// @author Ashish Kumar Meshram


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include <mmintrin.h>
#include <immintrin.h>
#include "utils.h"


// #define LOGG
// #define DEBUG_SSE
// #define DEBUG_AVX256
// #define DEBUG_AVX512

void* malloc_align(size_t N, size_t Nsize, size_t alignment) {    
    // Calculate total size needed for allocation
    size_t size = N*Nsize;

    // Allocate memory with specified alignment
    void* ptr = _mm_malloc(size, alignment);

    // Check if the memory has been successfully allocated
    if (ptr == NULL) {
        printf("Error: Memory allocation failed.\n");
        return NULL;
        exit(0); // Return error code
    
    // Memory has been successfully, now check if the memory has been 
    // successfully alligned
    } else if (((uintptr_t)ptr % alignment) == 1) {
        printf("ERROR: Memory is not aligned\n");
        return NULL;
        exit(0); // Return error code
    // Memory has been successfully alligned
    } else {
         printf("SUCCESS: %zu bytes of aligned memory allocated.\n", size);
         #ifdef LOGG
            // Print the starting address
            printf("Starting address: %p\n", (void*)ptr);

            // Calculate and print the ending address
            void* end_address = (void*)((char*)ptr + size - 1);
            printf("Ending address: %p\n", end_address);
         #endif
    }
    
    // Initialize allocated memory with zero
    memset(ptr, 0, size);

    return ptr;
}

void printf_intrinsic(char *s, char *dtype, int num, void *x) {
  if (strcmp(dtype, "int") == 0) {
    int *tempb = (int *)x;
    if (num == 128) {
      printf("%s: [%d, %d, %d, %d]\n", s, tempb[0], tempb[1],
             tempb[2], tempb[3]);
    } else if (num == 256) {
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d]\n", s,
             tempb[0], tempb[1], tempb[2], tempb[3],
             tempb[4], tempb[5], tempb[6], tempb[7]);
    } else if (num == 512) {
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n", s,
             tempb[0],  tempb[1],  tempb[2],  tempb[3],
             tempb[4],  tempb[5],  tempb[6],  tempb[7],
             tempb[8],  tempb[9],  tempb[10], tempb[11],
             tempb[12], tempb[13], tempb[14], tempb[15]);
    }
  }

  if (strcmp(dtype, "short") == 0) {
    short *tempb = (short *)x;
    if (num == 128)
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d]\n", s,
             tempb[0], tempb[1], tempb[2], tempb[3],
             tempb[4], tempb[5], tempb[6], tempb[7]);
    else if (num == 256)
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, \
                   %d]\n", s,
             tempb[0],  tempb[1],  tempb[2],  tempb[3],
             tempb[4],  tempb[5],  tempb[6],  tempb[7],
             tempb[8],  tempb[9],  tempb[10], tempb[11],
             tempb[12], tempb[13], tempb[14], tempb[15]);
    else if (num == 512)
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, \
                   %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, \
                   %d, %d]\n", s,
             tempb[0],  tempb[1],  tempb[2],  tempb[3],
             tempb[4],  tempb[5],  tempb[6],  tempb[7],
             tempb[8],  tempb[9],  tempb[10], tempb[11],
             tempb[12], tempb[13], tempb[14], tempb[15],
             tempb[16], tempb[17], tempb[18], tempb[19],
             tempb[20], tempb[21], tempb[22], tempb[23],
             tempb[24], tempb[25], tempb[26], tempb[27],
             tempb[28], tempb[29], tempb[30], tempb[31]);
  }
  
  if (strcmp(dtype, "char") == 0)
  {
    char *tempb = (char *)x;
    printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, \
                 %d]\n", s,
           tempb[0],  tempb[1],  tempb[2],  tempb[3], 
           tempb[4],  tempb[5],  tempb[6],  tempb[7],
           tempb[8],  tempb[9],  tempb[10], tempb[11], 
           tempb[12], tempb[13], tempb[14], tempb[15]);
  }
  if (strcmp(dtype, "float") == 0) {
    float *tempb = (float *)x;
    if (num == 128)
        printf("%s: [%f, %f, %f, %f]\n", s, 
              tempb[0], tempb[1], tempb[2], tempb[3]);
    else if (num == 256)
        printf("%s: [%f, %f, %f, %f, %f, %f, %f, %f]\n", s, tempb[0], tempb[1], 
                   tempb[2], tempb[3], tempb[4], tempb[5], tempb[6], tempb[7]);
    else if (num == 512)
        printf("%s: [%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, \
                    %f, %f]\n", s, 
                tempb[0],  tempb[1],  tempb[2],  tempb[3], 
                tempb[4],  tempb[5],  tempb[6],  tempb[7], 
                tempb[8],  tempb[9],  tempb[10], tempb[11], 
                tempb[12], tempb[13], tempb[14], tempb[15]);
    else
        printf("Number of bits not supported");
  }
  if (strcmp(dtype, "double") == 0)
  {
    double *tempb = (double *)x;
    printf("%s: [%f, %f]\n", s, tempb[0], tempb[1]);
  }
}

void print_array(char *dtype, void *x, char *s, int N1, int N2) {
    if (strcmp(dtype, "float") == 0) {
        float *xf = (float *)x;
        printf("Array %s between %d and %d are:\n", s, N1, N2 - 1);
        for (size_t k = N1; k < N2; k++) {
            if (xf[2*k + 1] > 0) {
                printf("%s[%lld] = %.4lf + 1i*%.4lf\n", s, k, xf[2*k], xf[2*k + 1]);
            } else {
                printf("%s[%lld] = %.4lf - 1i*%.4lf\n", s, k, xf[2*k], -1*xf[2*k + 1]);
            }
        }
    } else if (strcmp(dtype, "int16_t") == 0) {
        int16_t *xi = (int16_t *)x;
        printf("Array %s between %d and %d are:\n", s, N1, N2 - 1);
        for (size_t k = N1; k < N2; k++) {
            if (xi[2*k + 1] > 0) {
                printf("%s[%lld] = %d + 1i*%d\n", s, k, xi[2*k], xi[2*k + 1]);
            } else {
                printf("%s[%lld] = %d - 1i*%d\n", s, k, xi[2*k], -1*xi[2*k + 1]);
            }
        }
    }
}

void read_binary(void *ptr, char *dtype) {
    if (strcmp(dtype,"char") == 0) {    // read character from memory
        uint8_t *p2f = (uint8_t*)ptr;
        // 10000000
        uint16_t mask = 0x80;
        for (size_t i = 1; i <= sizeof(char)*8; i++) {
            putchar(*p2f & mask ? '1' : '0');
            mask >>= 1;
        }
        putchar('\n');
    }
    if (strcmp(dtype,"short") == 0) {    // read short from memory
        uint16_t *p2f = (uint16_t*)ptr;
        // 10000000 00000000
        uint16_t mask = 0x8000;
        for (size_t i = 1; i <= sizeof(short)*8; i++) {
            putchar(*p2f & mask ? '1' : '0');
            mask >>= 1;
            if (i % 8 == 0) { // output space after 8 bits
                putchar(' ');
            }
        }
        putchar('\n');
    }
    if (strcmp(dtype,"int") == 0) {    // read integer from memory
        uint32_t *p2f = (uint32_t*)ptr;
        // // 10000000 00000000 00000000 00000000
        uint32_t mask = 0x80000000;
        for (size_t i = 1; i <= sizeof(int)*8; i++) {
            putchar(*p2f & mask ? '1' : '0');
            mask >>= 1;
            if (i % 8 == 0) { // output space after 8 bits
                putchar(' ');
            }
        }
        putchar('\n');
    }
    if (strcmp(dtype,"float") == 0) {    // read float from memory
    // https://www.h-schmidt.net/FloatConverter/IEEE754.html
        uint32_t *p2f = (uint32_t*)ptr;
        // 10000000 00000000 00000000 00000000
        uint32_t mask = 0x80000000;
        for (size_t i = 1; i <= sizeof(float)*8; i++) {
            putchar(*p2f & mask ? '1' : '0');
            mask >>= 1;
            if (i == 1) {
                putchar(' ');
            }
            else if (i == 9) {
                putchar(' ');
            }
        }
        putchar('\n');
    }
    if (strcmp(dtype,"double") == 0) {    // read double from memory
        uint64_t *p2f = (uint64_t*)ptr;
        // 10000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
        uint64_t mask = 0x8000000000000000;
        for (size_t i = 1; i <= sizeof(double)*8; i++) {
            putchar(*p2f & mask ? '1' : '0');
            mask >>= 1;
            if (i == 1) {
                putchar(' ');
            }
            else if (i == 12) {
                putchar(' ');
            }
        }
        putchar('\n');
    }
}

void unif_randf(float a, float b, float *c, int N) {
    float range = b - a;
    for (size_t k = 0; k < N; k++) {
        float rand_real = ((float)rand() / RAND_MAX) * range + a;
        float rand_imag = ((float)rand() / RAND_MAX) * range + a;
        c[2 * k]     = rand_real;
        c[2 * k + 1] = rand_imag;
    }
}

int16_t fp_to_q15(float xfp) {
    int16_t xFP;
    if (xfp < -1)
        xFP = 0x8000;     // 32768 = 0b1000,0000,0000,0000
    else if (xfp >= 1)
        xFP = 0x7FFF;     // 37267 = 0b0111,1111,1111,1111
    else
        xFP = xfp*32768;
    return xFP;
}

float q15_to_fp(int16_t xFP) {
    return (float)(xFP/32768.0);
}

void float_to_fixed_point(float *xfp, int16_t *xFP, int N, float maxval, int scval) {
    for (size_t k = 0; k < N; k++) {
        // Scale the floating-point values
        float scaled_x = xfp[2*k] / (scval * maxval);
        float scaled_y = xfp[2*k + 1] / (scval * maxval);

        // Convert the scaled floating-point values to fixed-point representation
        xFP[2*k]     = fp_to_q15(scaled_x);
        xFP[2*k + 1] = fp_to_q15(scaled_y);
    }
}

void fixed_point_to_float(int16_t *xFP, float *xfp, int N, float maxval, int scval) {
    for (size_t k = 0; k < N; k++) {
        // Convert fixed-point values to floating-point representation
        float fp_x = q15_to_fp(xFP[2*k]);
        float fp_y = q15_to_fp(xFP[2*k + 1]);

        // Rescale the floating-point values
        xfp[2*k]     = fp_x * scval * scval * maxval * maxval;
        xfp[2*k + 1] = fp_y * scval * scval * maxval * maxval;
    }
}

float calculateMSE(float *x, float *y, int N) {
    float sum_squared_diff = 0.0f;

    for (size_t k = 0; k < N; k++) {
        float diff = x[k] - y[k];
        sum_squared_diff += diff * diff;
    }

    float mse = sum_squared_diff / N;
    return mse;
}

void complex_mult_float(float *x, float *y, float *z, int N) {
    for (size_t k = 0; k < N; k++) {
        // Extract real and imaginary parts from x and y
        float x_real = x[2*k];
        float x_imag = x[2*k + 1];
        float y_real = y[2*k];
        float y_imag = y[2*k + 1];
        
        // Perform complex multiplication
        z[2*k]     = x_real*y_real - x_imag*y_imag; // Real part
        z[2*k + 1] = x_real*y_imag + x_imag*y_real; // Imaginary part
    }

}

void complex_mult_fixed_point(int16_t *x, int16_t *y, int16_t *z, int N) {
    for (size_t k = 0; k < N; k++) {
        // Extract real and imaginary parts from x and y
        int16_t x_real = x[2*k];
        int16_t x_imag = x[2*k + 1];
        int16_t y_real = y[2*k];
        int16_t y_imag = y[2*k + 1];
        
        // Perform fixed-point multiplication
        int32_t z_real_tmp = ((int32_t)x_real * y_real) - ((int32_t)x_imag * y_imag);
        int32_t z_imag_tmp = ((int32_t)x_real * y_imag) + ((int32_t)x_imag * y_real);

        // Shift right to truncate to Q15 format
        int16_t z_real = z_real_tmp >> 15;
        int16_t z_imag = z_imag_tmp >> 15;

        // Store results in z
        z[2*k]     = z_real;
        z[2*k + 1] = z_imag;
    }
}

void complex_mult_float_SSE(float *x, float *y, float *z, int N) {
    __m128 x_real, x_imag, y_real, y_imag, z_real, z_imag;
    __m128 result_real, result_imag;

    for (size_t k = 0; k < N; k += 4) {
        // 
        x_real = _mm_set_ps(x[2*k + 6], x[2*k + 4], x[2*k + 2], x[2*k]);
        #ifdef DEBUG_SSE
            printf_intrinsic("x_real", "float", 128, &x_real);
        #endif
        x_imag = _mm_set_ps(x[2*k + 7], x[2*k + 5], x[2*k + 3], x[2*k + 1]);
        #ifdef DEBUG_SSE
            printf_intrinsic("x_imag", "float", 128, &x_imag);
        #endif
        y_real = _mm_set_ps(y[2*k + 6], y[2*k + 4], y[2*k + 2], y[2*k]);
        #ifdef DEBUG_SSE
            printf_intrinsic("y_real", "float", 128, &y_real);
        #endif
        y_imag = _mm_set_ps(y[2*k + 7], y[2*k + 5], y[2*k + 3], y[2*k + 1]);
        #ifdef DEBUG_SSE
            printf_intrinsic("y_imag", "float", 128, &y_imag);
        #endif

        // Perform complex multiplication
        result_real = _mm_sub_ps(_mm_mul_ps(x_real, y_real), _mm_mul_ps(x_imag, y_imag));
        #ifdef DEBUG_SSE
            printf_intrinsic("result_real", "float", 128, &result_real);
        #endif
        result_imag = _mm_add_ps(_mm_mul_ps(x_real, y_imag), _mm_mul_ps(x_imag, y_real));
        #ifdef DEBUG_SSE
            printf_intrinsic("result_imag", "float", 128, &result_imag);
        #endif

        // Interleave real and imaginary parts
        z_real = _mm_unpacklo_ps(result_real, result_imag);
        #ifdef DEBUG_SSE
            printf_intrinsic("z_real", "float", 128, &z_real);
        #endif
        z_imag = _mm_unpackhi_ps(result_real, result_imag);
        #ifdef DEBUG_SSE
            printf_intrinsic("z_real", "float", 128, &z_imag);
        #endif

        // Store the result in interleaved fashion
        _mm_store_ps(&z[2*k], z_real);
        _mm_store_ps(&z[2*k + 4], z_imag);
    }
}

void complex_mult_float_AVX256(float *x, float *y, float *z, int N) {
    __m256 x_real, x_imag, y_real, y_imag, z_real, z_imag;
    __m256 result_real, result_imag;

    for (size_t k = 0; k < N; k += 8) {
        // Load complex numbers from memory
        // Load four SPFP values from x
        x_real = _mm256_set_ps(x[2*k + 14], x[2*k + 12], x[2*k + 10], x[2*k + 8], 
                               x[2*k + 6],  x[2*k + 4],  x[2*k + 2],  x[2*k]);
        #ifdef DEBUG_AVX256
            printf_intrinsic("x_real", "float", 256, &x_real);
        #endif
        x_imag = _mm256_set_ps(x[2*k + 15], x[2*k + 13], x[2*k + 11], x[2*k + 9],
                               x[2*k + 7],  x[2*k + 5],  x[2*k + 3],  x[2*k + 1]);
        #ifdef DEBUG_AVX256
            printf_intrinsic("x_imag", "float", 256, &x_imag);
        #endif
        y_real = _mm256_set_ps(y[2*k + 14], y[2*k + 12], y[2*k + 10], y[2*k + 8],
                               y[2*k + 6],  y[2*k + 4],  y[2*k + 2],  y[2*k]);
        #ifdef DEBUG_AVX256
            printf_intrinsic("y_real", "float", 256, &y_real);
        #endif
        y_imag = _mm256_set_ps(y[2*k + 15], y[2*k + 13], y[2*k + 11], y[2*k + 9],
                               y[2*k + 7],  y[2*k + 5],  y[2*k + 3],  y[2*k + 1]);
        #ifdef DEBUG_AVX256
            printf_intrinsic("y_imag", "float", 256, &y_imag);
        #endif

        // Perform complex multiplication
        result_real = _mm256_sub_ps(_mm256_mul_ps(x_real, y_real), _mm256_mul_ps(x_imag, y_imag));
        #ifdef DEBUG_AVX256
            printf_intrinsic("result_real", "float", 256, &result_real);
        #endif
        result_imag = _mm256_add_ps(_mm256_mul_ps(x_real, y_imag), _mm256_mul_ps(x_imag, y_real));
        #ifdef DEBUG_AVX256
            printf_intrinsic("result_imag", "float", 256, &result_imag);
        #endif
        
        __m256i control_mask1 = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
        __m256 realperm1 = _mm256_permutevar8x32_ps(result_real, control_mask1);
        __m256 imagperm1 = _mm256_permutevar8x32_ps(result_imag, control_mask1);
        
        // Interleave real and imaginary parts
        z_real = _mm256_unpacklo_ps(realperm1, imagperm1);
        #ifdef DEBUG_AVX256
            printf_intrinsic("z_real", "float", 256, &z_real);
        #endif
        z_imag = _mm256_unpackhi_ps(realperm1, imagperm1);
        #ifdef DEBUG_AVX256
            printf_intrinsic("z_imag", "float", 256, &z_imag);
        #endif

        // Store the result in interleaved fashion
        _mm256_store_ps(&z[2*k], z_real);
        _mm256_store_ps(&z[2*k + 8], z_imag);
    }
}

void complex_mult_fixed_point_SSE(int16_t *x, int16_t *y, int16_t *z, int N, int shift) {
    // Casting x, y, z to __m128i
    __m128i *x128i = (__m128i *) x;
    __m128i *y128i = (__m128i *) y;
    __m128i *z128i = (__m128i *) z;

    // Intermediate variables
    __m128i c128i, re128i, im128i, lo128i, hi128i;

    c128i = _mm_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1);
    #ifdef DEBUG_SSE
       printf_intrinsic("c128i", "short", 128, &c128i);
    #endif

    for (size_t k = 0; k < N>>2; k++) {
        // Sign
        re128i = _mm_sign_epi16(x128i[k], c128i);
        #ifdef DEBUG_SSE
            printf_intrinsic("x128i", "short", 128, &x128i[k]);
            printf_intrinsic("re128i", "short", 128, &re128i);
        #endif
        // Multiply and add
        re128i = _mm_madd_epi16(y128i[k], re128i);
        // Shuffle Lower
        im128i = _mm_shufflelo_epi16(y128i[k], _MM_SHUFFLE(2,3,0,1));
        // Shuffle Higher
        im128i = _mm_shufflehi_epi16(im128i, _MM_SHUFFLE(2,3,0,1));
        // Multiply and add
        im128i = _mm_madd_epi16(x128i[k], im128i);
        // Shift Right
        re128i = _mm_srai_epi32(re128i, shift);
        // Shift Right
        im128i = _mm_srai_epi32(im128i, shift);
        // Unpack Lower
        lo128i = _mm_unpacklo_epi32(re128i, im128i);
        // Unpack Higher
        hi128i = _mm_unpackhi_epi32(re128i, im128i);
        // Pack
        z128i[k]  = _mm_packs_epi32(lo128i, hi128i);
        // Increment Pointer
        x128i[0] = x128i[0] + 8;
        y128i[0] = y128i[0] + 8;
        z128i[0] = z128i[0] + 8;
    }
}

void complex_mult_fixed_point_AVX256(int16_t *x, int16_t *y, int16_t *z, int N, int shift) {
     // Casting x, y, z to __m256i
    __m256i *x256i = (__m256i *) x;
    __m256i *y256i = (__m256i *) y;
    __m256i *z256i = (__m256i *) z;
    __m256i re256i, im256i, lo256i, hi256i;
    __m256i conj = _mm256_set_epi16(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);
    
    for (size_t k = 0; k < N>>3; k++) {
        // Sign
        re256i = _mm256_sign_epi16(x256i[k], conj);
        // Multiply and add
        re256i = _mm256_madd_epi16(y256i[k], re256i);
        // Shuffle Lower
        im256i = _mm256_shufflelo_epi16(y256i[k], _MM_SHUFFLE(2,3,0,1));
        // Shuffle Higher
        im256i = _mm256_shufflehi_epi16(im256i, _MM_SHUFFLE(2,3,0,1));
        // Multiply and add
        im256i = _mm256_madd_epi16(x256i[k], im256i);
        // Shift Right
        re256i = _mm256_srai_epi32(re256i, shift);
        // Shift Right
        im256i = _mm256_srai_epi32(im256i, shift);
        // Unpack Lower
        lo256i = _mm256_unpacklo_epi32(re256i, im256i);
        // Unpack Higher
        hi256i = _mm256_unpackhi_epi32(re256i, im256i);
        // Pack
        z256i[k] = _mm256_packs_epi32(lo256i, hi256i);
        // Increment Pointer
        x256i[0] = x256i[0] + 16;
        y256i[0] = y256i[0] + 16;
        z256i[0] = z256i[0] + 16;
    }
}

void write_iq_to_mfile(const char *filename, const char *varname, float *c, int N) {

    if (filename == NULL || varname == NULL) {
        printf("Error: filename or varname is NULL\n");
        return;
    }

    if (N < 0) {
        printf("Error: Number of elements cannot be negative\n");
        return;
    }
    FILE *fp = NULL;
    fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file for writing\n");
        return;
    }

    // Write complex numbers into MATLAB script file
    fprintf(fp, "iqsamples = [");
    for (int i = 0; i < N; i++) {
        if (c[2 * i + 1] > 0) {
            fprintf(fp, "%.4lf + 1i*%.4lf", c[2 * i], c[2 * i + 1]);
        } else {
            fprintf(fp, "%.4lf - 1i*%.4lf", c[2 * i], -1*c[2 * i + 1]);
        }
        if (i < N - 1) {
            fprintf(fp, "; ...\n ");
        }
    }
    fprintf(fp, "];\n");    

    fprintf(fp, "save('%s.mat', 'iqsamples');\n", varname);

    // Close the file
    fclose(fp);
}

void write_to_mfile(const char *filename, const char *varname, float *c, int N) {

    if (filename == NULL || varname == NULL) {
        printf("Error: filename or varname is NULL\n");
        return;
    }

    if (N < 0) {
        printf("Error: Number of elements cannot be negative\n");
        return;
    }
    FILE *fp = NULL;
    fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return;
    }

    // Write the array to the file with the specified variable name
    fprintf(fp, "%s = [", varname);
    for (size_t k = 0; k < N - 1; k++) {
        fprintf(fp, "%.8f, ", c[k]);
    }
    fprintf(fp, "%.8f];\n", c[N - 1]); // Last value without comma

    fprintf(fp, "save('%s.mat', '%s');\n", varname, varname);

    // Close the file
    fclose(fp);
}

void extract_element_from_array(float *x, int N, float *y, int M) {
   // Check if M is less than N
    if (M >= N) {
        printf("Error: M should be less than N\n");
        return;
    }
    memcpy(y, x, M*sizeof(float));
}