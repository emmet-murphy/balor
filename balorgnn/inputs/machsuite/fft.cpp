
#include <stdio.h>
#include <stdlib.h>

#define FFT_SIZE 1024
#define twoPI 6.28318530717959


void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE / 2], double img_twid[FFT_SIZE / 2]) {
    int even, odd, span, log, rootindex;
    double temp;
    log = 0;

    outer:for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        #pragma HLS TRIPCOUNT AVG=10
        inner:for (odd = span; odd < FFT_SIZE; odd++) {
            #pragma HLS TRIPCOUNT AVG=921.7
            odd |= span;
            even = odd ^ span;

            temp = real[even] + real[odd];
            real[odd] = real[even] - real[odd];
            real[even] = temp;

            temp = img[even] + img[odd];
            img[odd] = img[even] - img[odd];
            img[even] = temp;

            rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                temp = real_twid[rootindex] * real[odd] - img_twid[rootindex] * img[odd];
                img[odd] = real_twid[rootindex] * img[odd] + img_twid[rootindex] * real[odd];
                real[odd] = temp;
            }
        }
    }
}
