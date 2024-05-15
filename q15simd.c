/// @author Ashish Kumar Meshram


#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>   
#include <emmintrin.h>
#include <immintrin.h>
#include "utils.h"

#define ALIGNMENT 32 // Align to 32 bytes for SSE/AVX256 instructions

int main(int argc, char *argv[]) {

    int test_flag   = 0;  // Flag to track if -test flag is provided
    int export_flag = 1;  // Flag for exporting data to m-files
    int logg        = 1;  // Flag to display information

    // Default parameters (can be updated during execution)
    int Nsamp    = 8;          // Number of IQ samples to be generated
    float maxval = 10.42659;   // Maximum absolute value
    int scalval  = 2;          // Scaling value for Q15 format

    if(argc < 2) {
            printf("[ERROR] Missing flag.\n");
            return 1; // Exit with error code
    }
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-test") == 0) {
            if (test_flag) {
                // If -test flag is already provided
                printf("[ERROR] Multiple instances of -test flag.\n");
                return 1; // Exit with error code
            } else {
                test_flag = 1; // Set test_flag
            }
        } else if (test_flag) {
            // If -test flag is provided, don't process other flags
            printf("[ERROR] Cannot provide other flags with -test.\n");
            return 1; // Exit with error code
        } else if (strcmp(argv[i], "-NSample") == 0 && i + 1 < argc) {
            Nsamp = atoi(argv[i + 1]);
            // Check if Nsamp is divisible by 8
            if (Nsamp % 8 != 0) {
                printf("Error: Nsamp should be divisible by 8.\n");
                return 1; // Exit with error code
            } else if (Nsamp == 8) {        
                export_flag = 1;
                logg = 1;
            } else {
                export_flag = 0;
                logg = 0;
            }
            i++;  // Skip the value argument
        } else if (strcmp(argv[i], "-MaxVal") == 0 && i + 1 < argc) {
            maxval = atof(argv[i + 1]);
            i++;  // Skip the value argument
        } else if (strcmp(argv[i], "-Scale") == 0 && i + 1 < argc) {
            scalval = atoi(argv[i + 1]);
            i++;  // Skip the value argument
        } else {
            // If any other flag is provided
            printf("[ERROR] Invalid flag.\n");
            return 1; // Exit with error code
        }
    }
    
    // Print information provided
    printf("[INFO] %d complex samples to be processed\n", Nsamp);
    printf("[INFO] Range: [%f, %f]\n", -fabs(maxval), maxval);
    printf("[INFO] Scaling factor = %d\n", scalval);

    // Dynamically allocate memory with alignment and initialize with zero
    float *xfp   = (float*)malloc_align(2*Nsamp, sizeof(float), ALIGNMENT);
    int16_t *xFP = (int16_t*)malloc_align(2*Nsamp, sizeof(int16_t), ALIGNMENT);
    float *yfp   = (float*)malloc_align(2*Nsamp, sizeof(float), ALIGNMENT);
    int16_t *yFP = (int16_t*)malloc_align(2*Nsamp, sizeof(int16_t), ALIGNMENT);

    float *zfp_sisd   = (float*)malloc_align(2*Nsamp, sizeof(float), ALIGNMENT);
    int16_t *zFP_SISD = (int16_t*)malloc_align(2*Nsamp, sizeof(int16_t), ALIGNMENT);
    float *zfp_SISD   = (float*)malloc_align(2*Nsamp, sizeof(float), ALIGNMENT);

    float *zfp_simd_sse   = (float*)malloc_align(2*Nsamp, sizeof(float), ALIGNMENT);
    int16_t *zFP_SIMD_SSE = (int16_t*)malloc_align(2*Nsamp, sizeof(int16_t), ALIGNMENT);
    float *zfp_SIMD_SSE   = (float*)malloc_align(2*Nsamp, sizeof(float), ALIGNMENT);

    float *zfp_simd_avx256   = (float*)malloc_align(2*Nsamp, sizeof(float), ALIGNMENT);
    int16_t *zFP_SIMD_AVX256 = (int16_t*)malloc_align(2*Nsamp, sizeof(int16_t), ALIGNMENT);
    float *zfp_SIMD_AVX256   = (float*)malloc_align(2*Nsamp, sizeof(float), ALIGNMENT);

    // Few more parameters
    float LL  = -fabs(maxval);  // Lower limit
    float UL  = maxval;         // Upper limit    
    int shift = 15;             // Bit shift value for Q15 format

    // Generate random IQ samples
    srand(time(NULL));              // Seed the random number generator
    unif_randf(LL, UL, xfp, Nsamp); // Store into xfp
    if(logg) {
        printf("Generating unifromly distributed complex random samples\n");
    }
    // Generate random IQ samples
    unif_randf(LL, UL, yfp, Nsamp); // Store into yfp
    if(logg) {
        printf("Generating unifromly distributed complex random samples\n");
    }
    
    // Convert to Q15 number
    float_to_fixed_point(xfp, xFP, Nsamp, maxval, scalval); // xFP = Q15(xfp)
    float_to_fixed_point(yfp, yFP, Nsamp, maxval, scalval); // yFP = Q15(yfp)

    // --- Perform single precision floating point complex multiplication --- //
    clock_t start_fpsisd = clock();
    complex_mult_float(xfp, yfp, zfp_sisd, Nsamp);         // zfp_sisd = xfp*yfp
    if(logg) {
        printf("Computed single precision floating point complex multiplication\n");
    }
    clock_t end_fpsisd = clock();
    double fpsisd_CPU_TIME = ((float)(end_fpsisd - start_fpsisd)) / CLOCKS_PER_SEC;

    clock_t start_fpsse = clock();
    complex_mult_float_SSE(xfp, yfp, zfp_simd_sse, Nsamp);  // zfp_simd_sse = xfp*yfp
    if(logg) {
        printf("Computed single precision floating point complex multiplication using SSE\n");
    }
    clock_t end_fpsse = clock();
    double fpsse_CPU_TIME = ((float)(end_fpsse - start_fpsse)) / CLOCKS_PER_SEC;

    clock_t start_fpavx256 = clock();
    complex_mult_float_AVX256(xfp, yfp, zfp_simd_avx256, Nsamp);    // zfp_simd_avx256 = xfp*yfp
    if(logg) {
        printf("Computed single precision floating point complex multiplication using AVX256\n");
    }
    clock_t end_fpavx256 = clock();
    double fpavx256_CPU_TIME = ((float)(end_fpavx256 - start_fpavx256)) / CLOCKS_PER_SEC;

    // ----------- Perform Q15 fixed point complex multiplication ----------- //
    clock_t start_q15 = clock();
    complex_mult_fixed_point(xFP, yFP, zFP_SISD, Nsamp);    // zFP_SISD = xFP*yFP
    if(logg) {
        printf("Computed Q15 fixed point complex multiplication\n");
    }
    clock_t end_q15 = clock();
    double q15_CPU_TIME = ((float)(end_q15 - start_q15)) / CLOCKS_PER_SEC;

    clock_t start_q15sse = clock();
    complex_mult_fixed_point_SSE(xFP, yFP, zFP_SIMD_SSE, Nsamp, shift); // zFP_SIMD_SSE = xFP*yFP
    if(logg) {
        printf("Computed Q15 fixed point complex multiplication using SSE\n");
    }
    clock_t end_q15sse = clock();
    double q15sse_CPU_TIME = ((float)(end_q15sse - start_q15sse)) / CLOCKS_PER_SEC;

    clock_t start_q15avx256 = clock();
    complex_mult_fixed_point_AVX256(xFP, yFP, zFP_SIMD_AVX256, Nsamp, shift);  // zFP_SIMD_AVX256 = xFP*yFP
    if(logg) {
        printf("Computed Q15 fixed point complex multiplication using AVX256\n");
    }
    clock_t end_q15avx256 = clock();
    double q15avx256_CPU_TIME = ((float)(end_q15avx256 - start_q15avx256)) / CLOCKS_PER_SEC;

    // Convert Q15 number to single precision floating point number
    fixed_point_to_float(zFP_SISD, zfp_SISD, Nsamp, maxval, scalval);               // zfp_SISD = Q15_1(zFP_SISD)
    fixed_point_to_float(zFP_SIMD_SSE, zfp_SIMD_SSE, Nsamp, maxval, scalval);       // zfp_SIMD_SSE = Q15_1(zFP_SIMD_SSE)
    fixed_point_to_float(zFP_SIMD_AVX256, zfp_SIMD_AVX256, Nsamp, maxval, scalval); // zfp_SIMD_AVX256 = Q15_1(zFP_SIMD_AVX256)
    
    if(logg) {
        print_array("float", zfp_sisd       , "zfp"            , 0, Nsamp);
        print_array("float", zfp_simd_sse   , "zfp_simd_sse"   , 0, Nsamp);
        print_array("float", zfp_simd_avx256, "zfp_simd_avx256", 0, Nsamp);
        print_array("float", zfp_SISD       , "zfp_SISD"       , 0, Nsamp);
        print_array("float", zfp_SIMD_SSE   , "zfp_SIMD_SSE"   , 0, Nsamp);
        print_array("float", zfp_SIMD_AVX256, "zfp_SIMD_AVX256", 0, Nsamp);
    }

    // Compute MSE
    float mse_simd_sse    = calculateMSE(zfp_sisd, zfp_simd_sse   , 2*Nsamp);
    float mse_simd_avx256 = calculateMSE(zfp_sisd, zfp_simd_avx256, 2*Nsamp);
    float mse_SISD        = calculateMSE(zfp_sisd, zfp_SISD       , 2*Nsamp);
    float mse_SIMD_SSE    = calculateMSE(zfp_sisd, zfp_SIMD_SSE   , 2*Nsamp);
    float mse_SIMD_AVX256 = calculateMSE(zfp_sisd, zfp_SIMD_AVX256, 2*Nsamp);

    // Collect MSE in mse
    float mse[5] = {mse_simd_sse, mse_simd_avx256, mse_SISD, mse_SIMD_SSE,
                    mse_SIMD_AVX256};

    if(logg) {
        printf("Mean square error, MSE_simd_sse    = %f\n", mse_simd_sse);
        printf("Mean square error, MSE_simd_avx256 = %f\n", mse_simd_avx256);
        printf("Mean square error, MSE_SISD        = %f\n", mse_SISD);
        printf("Mean square error, MSE_SIMD_SSE    = %f\n", mse_SIMD_SSE);
        printf("Mean square error, MSE_SIMD_AVX256 = %f\n", mse_SIMD_AVX256);
    }

    // Collect CPU time
    float cputime[6] = {fpsisd_CPU_TIME, fpsse_CPU_TIME, fpavx256_CPU_TIME, 
                        q15_CPU_TIME, q15sse_CPU_TIME, q15avx256_CPU_TIME};

    // Export IQ samples, MSE, and CPU time metrics to MATLAB m-file
    if (export_flag) {
        // Initialize filename and variables for writing IQ sample and data into m-file
        const char *fn_xfp             = "xfp.m"             , *vn_xfp              = "xfp";
        const char *fn_yfp             = "yfp.m"             , *vn_yfp              = "yfp";
        const char *fn_zfp_sisd        = "zfp_sisd.m"        , *vn_zfp_sisd         = "zfp_sisd";
        const char *fn_zfp_SISD        = "zfp_SISD1.m"       , *vn_zfp_SISD1        = "zfp_SISD1";
        const char *fn_zfp_simd_sse    = "zfp_simd_sse.m"    , *vn_zfp_simd_sse     = "zfp_simd_sse";
        const char *fn_zfp_SIMD_SSE    = "zfp_SIMD1_SSE.m"   , *vn_zfp_SIMD1_SSE    = "zfp_SIMD1_SSE";
        const char *fn_zfp_simd_avx256 = "zfp_simd_avx256.m" , *vn_zfp_simd_avx256  = "zfp_simd_avx256";
        const char *fn_zfp_SIMD_AVX256 = "zfp_SIMD1_AVX256.m", *vn_zfp_SIMD1_AVX256 = "zfp_SIMD1_AVX256";
        const char *fn_mse             = "MSE.m"             , *vn_mse              = "mse";
        const char *fn_cpu_time        = "CPUtime.m"         , *vn_cpu_time         = "cpu_time";
        // Write complex numbers and data to MATLAB script file
        write_iq_to_mfile(fn_xfp, vn_xfp, xfp, Nsamp);
        printf("Wrote IQ samples to file named as %s \n", fn_xfp);
        write_iq_to_mfile(fn_yfp, vn_yfp, yfp, Nsamp);
        printf("Wrote IQ samples to file named as %s \n", fn_yfp);
        write_iq_to_mfile(fn_zfp_sisd, vn_zfp_sisd, zfp_sisd, Nsamp);
        printf("Wrote IQ samples to file named as %s \n", fn_zfp_sisd);
        write_iq_to_mfile(fn_zfp_SISD, vn_zfp_SISD1, zfp_SISD, Nsamp);
        printf("Wrote IQ samples to file named as %s \n", fn_zfp_SISD);
        write_iq_to_mfile(fn_zfp_simd_sse, vn_zfp_simd_sse, zfp_simd_sse, Nsamp);
        printf("Wrote IQ samples to file named as %s \n", fn_zfp_simd_sse);
        write_iq_to_mfile(fn_zfp_SIMD_SSE, vn_zfp_SIMD1_SSE, zfp_SIMD_SSE, Nsamp);
        printf("Wrote IQ samples to file named as %s \n", fn_zfp_SIMD_SSE);
        write_iq_to_mfile(fn_zfp_simd_avx256, vn_zfp_simd_avx256, zfp_simd_avx256, Nsamp);
        printf("Wrote IQ samples to file named as %s \n", fn_zfp_simd_avx256);
        write_iq_to_mfile(fn_zfp_SIMD_AVX256, vn_zfp_SIMD1_AVX256, zfp_SIMD_AVX256, Nsamp);
        printf("Wrote IQ samples to file named as %s \n", fn_zfp_SIMD_AVX256);
        write_to_mfile(fn_mse, vn_mse, mse, 5);
        printf("Wrote mse values to file named as %s \n", fn_mse);
        write_to_mfile(fn_cpu_time, vn_cpu_time, cputime, 6);
        printf("Wrote cpu time values to file named as %s \n", fn_cpu_time);
    } else {
        // Initialize filename and variables for writing data into m-file
        const char *fn_mse      = "MSE.m"    , *vn_mse      = "mse";
        const char *fn_cpu_time = "CPUtime.m", *vn_cpu_time = "cpu_time";
        // Write data to MATLAB script file
        write_to_mfile(fn_mse, vn_mse, mse, 5);
        printf("Wrote mse values to file named as %s \n", fn_mse);
        write_to_mfile(fn_cpu_time, vn_cpu_time, cputime, 6);
        printf("Wrote cpu time values to file named as %s \n", fn_cpu_time);
    }

    // Releasing dynamically allocated memory
    _mm_free(xfp);
    _mm_free(xFP);
    _mm_free(yfp);
    _mm_free(yFP);
    _mm_free(zfp_sisd);
    _mm_free(zFP_SISD);
    _mm_free(zfp_SISD);
    _mm_free(zfp_simd_sse);
    _mm_free(zFP_SIMD_SSE);
    _mm_free(zfp_SIMD_SSE);
    _mm_free(zfp_simd_avx256);
    _mm_free(zFP_SIMD_AVX256);
    _mm_free(zfp_SIMD_AVX256);
    
    printf("Execute Test1 or Test2 Matlab files for results!\n");
    // Return success
    return 0;
}