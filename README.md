# q15-simd



For compiling
`gcc -g utils.c q15simd.c -mavx2 -msse -lm -Wall -o q15simd`

For excution
1. `./q15simd -test`, with this program executes with the default parameters
2. `./q15simd -NSample 8000000 -MaxVal 12.4286 -Scale 4`
	where the flag are defined below as
	* `-NSample` specifies the number of complex samples will be 		generated during the execution
	* `-MaxVal` specifies the maximum value used for the 		generation of unifromly distriubted random number between $-		\text{MaxVal}$ to $-\text{MaxVal}$
	* `-Scale` specifies the value used while converting single 		precision floating point to Q-15 fixed point number.
