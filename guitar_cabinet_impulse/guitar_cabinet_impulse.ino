/*
 * (c) DD4WH 09/11/2019
 * 
 * Real Time PARTITIONED BLOCK CONVOLUTION FILTERING (STEREO)
 * 
 * thanks a lot to Brian Millier and Warren Pratt for your help!
 * 
 * using a guitar cabinet impulse response with up to about 20000 coefficients per channel
 * 
 * uses Teensy 4.0 and external ADC / DAC connected on perf board with ground plane
 * should be able to run with the Teensy audio shield rev D (although not tested)
 * 
 *  PCM5102A DAC module
    VCC = Vin
    3.3v = NC
    GND = GND
    FLT = GND
    SCL = 23 / MCLK via series 100 Ohm
    BCK = BCLK (21)
    DIN = TX (7)
    LCK = LRCLK (20)
    FMT = GND
    XMT = 3.3V (HIGH)
    
    PCM1808 ADC module:    
    FMT = GND
    MD1 = GND
    MD0 = GND
    GND = GND
    3.3V = 3.3V --> ADC needs both: 5V AND 3V3
    5V = VIN
    BCK = BCLK (21) via series 100 Ohm
    OUT = RX (8)
    LRC = LRCLK (20) via series 100 Ohm
    SCK = MCLK (23) via series 100 Ohm
    GND = GND
    3.3V = 3.3V
 *  
 * inspired by and uses code from wdsp library by Warren Pratt
 * https://github.com/g0orx/wdsp/blob/master/firmin.c
 * 
 * in the public domain GNU GPL v3
 */

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <arm_math.h>
#include <arm_const_structs.h> // in the Teensy 4.0 audio library, the ARM CMSIS DSP lib is already a newer version
#include <utility/imxrt_hw.h> // necessary for setting the sample rate, thanks FrankB !

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// USER DEFINES

// Choose only one of these impulse responses !

//#define IR1  // 512 taps // MG impulse response from bmillier github @44.1ksps
//#define IR2  // 4096 taps // impulse response @44.1ksps
//#define IR3  // 7552 taps // impulse response @44.1ksps
//#define IR4    // 17920 taps // impulse response 400ms @44.1ksps
//#define IR5    // 21632 taps // impulse response 490ms @44.1ksps
//#define IR6 // 5760 taps, 18.72% load vs. 15.00%
#define IR7 // 22016 taps, 50.62% load, 93.48% RAM1, 32064 bytes free
//#define IR8 // 25552 taps, too much !
// about 25000 taps is MAXIMUM --> about 0.5 seconds
//#define LPMINPHASE512 // 512 taps minimum phase 2.7kHz lowpass filter
//#define LPMINPHASE1024 // 1024 taps minimum phase 2.7kHz lowpass filter
//#define LPMINPHASE2048PASSTHRU // 2048 taps minimum phase 19.0kHz lowpass filter
//#define LPMINPHASE4096 // 4096 taps minimum phase 2.7kHz lowpass filter
const float32_t PROGMEM audio_gain = 1.0; // has to be adjusted from 1.0 to 10.0 depending on the filter gain / impulse resonse gain
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(IR1)
#include "impulse_response_1.h"
const int nc = 512; // number of taps for the FIR filter
#elif defined(IR2)
#include "impulse_response_2.h"
const int nc = 4096; // number of taps for the FIR filter
#elif defined(IR3)
#include "impulse_response_3.h"
const int nc = 7552; // number of taps for the FIR filter
#elif defined(IR5)
#include "impulse_response_5.h"
const int nc = 21632; // number of taps for the FIR filter
#elif defined(IR6)
#include "impulse_response_6.h"
const int nc = 5760; // number of taps for the FIR filter, 
#elif defined(IR7)
#include "impulse_response_7.h"
const int nc = 22016; // number of taps for the FIR filter, 
#elif defined(IR8)
#include "impulse_response_8.h"
const int nc = 25552; // number of taps for the FIR filter, 
#elif defined(LPMINPHASE512)
#include "lp_minphase_512.h"
const int nc = 512;
#elif defined(LPMINPHASE1024)
#include "lp_minphase_1024.h"
const int nc = 1024;
#elif defined(LPMINPHASE2048PASSTHRU)
#include "lp_minphase_2048passthru.h"
const int nc = 2048;
#elif defined(LPMINPHASE4096)
#include "lp_minphase_4096.h"
const int nc = 4096;
#else
#include "impulse_response_4.h"
const int nc = 17920; // number of taps for the FIR filter
#endif

extern "C" uint32_t set_arm_clock(uint32_t frequency);

#define LATENCY_TEST
const double PROGMEM FHiCut = 2500.0;
const double PROGMEM FLoCut = -FHiCut;
// define your sample rate
const double PROGMEM SAMPLE_RATE = 44100;  
// the latency of the filter is meant to be the same regardless of the number of taps for the filter
// partition size of 128 translates to a latency of 128/sample rate, ie. to 2.9msec with 44.1ksps

// latency can even be reduced by setting partitionsize to 64
// however, this only works, if you set AUDIO_BLOCK_SAMPLES to 64 in AudioStream.h
const int PROGMEM partitionsize = 128; 

#define DEBUG
#define FOURPI  (2.0 * TWO_PI)
#define SIXPI   (3.0 * TWO_PI)
#define BUFFER_SIZE partitionsize
int32_t sum;
int idx_t = 0;
int16_t *sp_L;
int16_t *sp_R;
uint8_t PROGMEM FIR_filter_window = 1;
const uint32_t PROGMEM FFT_L = 2 * partitionsize; 
float32_t mean = 1;
uint8_t first_block = 1; 
const uint32_t PROGMEM FFT_length = FFT_L;
const int PROGMEM nfor = nc / partitionsize; // number of partition blocks --> nfor = nc / partitionsize
//float DMAMEM cplxcoeffs[nc * 2]; // this holds the initial complex coefficients for the filter BEFORE partitioning
float32_t DMAMEM maskgen[FFT_L * 2];
float32_t DMAMEM fmask[nfor][FFT_L * 2]; // 
float32_t DMAMEM fftin[FFT_L * 2];
float32_t DMAMEM accum[FFT_L * 2];
float  fftout[nfor][FFT_L * 2]; // 

int buffidx = 0;
int k = 0;
//int idxmask = nfor - 1;

uint32_t all_samples_counter = 0;
uint8_t no_more_latency_test = 0;

const uint32_t N_B = FFT_L / 2 / BUFFER_SIZE;
uint32_t N_BLOCKS = N_B;
float32_t DMAMEM float_buffer_L [BUFFER_SIZE * N_B];  
float32_t DMAMEM float_buffer_R [BUFFER_SIZE * N_B]; 
float32_t DMAMEM last_sample_buffer_L [BUFFER_SIZE * N_B];  
float32_t DMAMEM last_sample_buffer_R [BUFFER_SIZE * N_B]; 
// complex FFT with the new library CMSIS V4.5
const static arm_cfft_instance_f32 *S;
// complex iFFT with the new library CMSIS V4.5
const static arm_cfft_instance_f32 *iS;
// FFT instance for direct calculation of the filter mask
// from the impulse response of the FIR - the coefficients
const static arm_cfft_instance_f32 *maskS;

// this audio comes from the codec by I2S
AudioInputI2S            i2s_in;
AudioRecordQueue         Q_in_L;
AudioRecordQueue         Q_in_R;
AudioMixer4              mixleft;
AudioMixer4              mixright;
AudioPlayQueue           Q_out_L;
AudioPlayQueue           Q_out_R;
AudioOutputI2S           i2s_out;

AudioConnection          patchCord1(i2s_in, 0, Q_in_L, 0);
AudioConnection          patchCord2(i2s_in, 1, Q_in_R, 0);

AudioConnection          patchCord3(Q_out_L, 0, mixleft, 0);
AudioConnection          patchCord4(Q_out_R, 0, mixright, 0);
AudioConnection          patchCord9(mixleft, 0,  i2s_out, 1);
AudioConnection          patchCord10(mixright, 0, i2s_out, 0);


void setup() {
  Serial.begin(115200);
  while(!Serial);

  AudioMemory(10); 
  delay(100);

  set_arm_clock(600000000);

  /****************************************************************************************
     Audio Setup
  ****************************************************************************************/
  mixleft.gain(0, 1.0);
  mixright.gain(0, 1.0);

  setI2SFreq(SAMPLE_RATE);

  /****************************************************************************************
     properly initialise variables in DMAMEM
  ****************************************************************************************/

  for(unsigned jj = 0; jj < nfor; jj++)
  {
    for(unsigned ii = 0; ii < FFT_L * 2; ii++)
    {
      fftout[jj][ii] = 0.1;
    }
  }

/*  for (unsigned i = 0; i < FFT_length * 2; i++)
  {
      cplxcoeffs[i] = 0.0;
  }
*/
  /****************************************************************************************
     set filter bandwidth
  ****************************************************************************************/
  // this routine does all the magic of calculating the FIR coeffs
  //calc_cplx_FIR_coeffs_interleaved (cplxcoeffs, nc, FLoCut, FHiCut, SAMPLE_RATE);
 // fir_bandpass (cplxcoeffs, nc, FLoCut, FHiCut, SAMPLE_RATE, 0, 1, 1.0);
//  fir_bandpass (cplxcoeffs, nc, FLoCut, FHiCut, SAMPLE_RATE, 0, 0, 1.0);

  /****************************************************************************************
     init complex FFTs
  ****************************************************************************************/
  switch (FFT_length)
  {
    case 128:
      S = &arm_cfft_sR_f32_len128;
      iS = &arm_cfft_sR_f32_len128;
      maskS = &arm_cfft_sR_f32_len128;
      break;
    case 256:
      S = &arm_cfft_sR_f32_len256;
      iS = &arm_cfft_sR_f32_len256;
      maskS = &arm_cfft_sR_f32_len256;
      break;
  }

  /****************************************************************************************
     Calculate the FFT of the FIR filter coefficients once to produce the FIR filter mask
  ****************************************************************************************/
    init_partitioned_filter_masks();

    flexRamInfo();

    Serial.println();
    Serial.print("AUDIO_BLOCK_SAMPLES:  ");     Serial.println(AUDIO_BLOCK_SAMPLES);

    Serial.println();
    
  /****************************************************************************************
     begin to queue the audio from the audio library
  ****************************************************************************************/
  delay(100);
  Q_in_L.begin();
  Q_in_R.begin();

} // END OF SETUP
  elapsedMillis msec = 0;
void loop() {
  elapsedMicros usec = 0;
  // are there at least N_BLOCKS buffers in each channel available ?
    if (Q_in_L.available() > N_BLOCKS + 0 && Q_in_R.available() > N_BLOCKS + 0)
    {

      // get audio samples from the audio  buffers and convert them to float
      for (unsigned i = 0; i < N_BLOCKS; i++)
      {
        sp_L = Q_in_L.readBuffer();
        sp_R = Q_in_R.readBuffer();

        // convert to float one buffer_size
        // float_buffer samples are now standardized from > -1.0 to < 1.0
        arm_q15_to_float (sp_L, &float_buffer_L[BUFFER_SIZE * i], BUFFER_SIZE); // convert int_buffer to float 32bit
        arm_q15_to_float (sp_R, &float_buffer_R[BUFFER_SIZE * i], BUFFER_SIZE); // convert int_buffer to float 32bit
        Q_in_L.freeBuffer();
        Q_in_R.freeBuffer();
      }
 
      /**********************************************************************************
          Digital convolution
       **********************************************************************************/
      //  basis for this was Lyons, R. (2011): Understanding Digital Processing.
      //  "Fast FIR Filtering using the FFT", pages 688 - 694
      //  numbers for the steps taken from that source
      //  Method used here: overlap-and-save

      // ONLY FOR the VERY FIRST FFT: fill first samples with zeros
      if (first_block) // fill real & imaginaries with zeros for the first BLOCKSIZE samples
      {
        for (unsigned i = 0; i < partitionsize * 4; i++)
        {
          fftin[i] = 0.0;
        }
        first_block = 0;
      }
      else
      {  // HERE IT STARTS for all other instances
        // fill FFT_buffer with last events audio samples
        for (unsigned i = 0; i < partitionsize; i++)
        {
          fftin[i * 2] = last_sample_buffer_L[i]; // real
          fftin[i * 2 + 1] = last_sample_buffer_R[i]; // imaginary
        }
      }
    
      // copy recent samples to last_sample_buffer for next time!
      for (unsigned i = 0; i < partitionsize; i++)
      {
        last_sample_buffer_L [i] = float_buffer_L[i];
        last_sample_buffer_R [i] = float_buffer_R[i];
      }

      // now fill recent audio samples into FFT_buffer (left channel: re, right channel: im)
      for (unsigned i = 0; i < partitionsize; i++)
      {
        fftin[FFT_length + i * 2] = float_buffer_L[i]; // real
        fftin[FFT_length + i * 2 + 1] = float_buffer_R[i]; // imaginary
      }

#if defined(LATENCY_TEST)
        if(msec > 2000 && !no_more_latency_test)
        {
        // latency test
        fftin[42] = 10.0; fftin[44] = -10.0;
        fftin[43] = 10.0; fftin[45] = -10.0;
        no_more_latency_test = 1;
        }
        if(no_more_latency_test == 1) all_samples_counter += partitionsize; 
#endif       
      /**********************************************************************************
          Complex Forward FFT
       **********************************************************************************/
      // calculation is performed in-place the FFT_buffer [re, im, re, im, re, im . . .]
      arm_cfft_f32(S, fftin, 0, 1);
      for(unsigned i = 0; i < partitionsize * 4; i++)
      {
          fftout[buffidx][i] = fftin[i];
      }

      /**********************************************************************************
          Complex multiplication with filter mask (precalculated coefficients subjected to an FFT)
          this is taken from wdsp library by Warren Pratt firmin.c
       **********************************************************************************/
      k = buffidx;

      for(unsigned i = 0; i < partitionsize * 4; i++)
      {
          accum[i] = 0.0;
      }
      
      for(unsigned j = 0; j < nfor; j++)
      { 
          for(unsigned i = 0; i < 2 * partitionsize; i= i + 4 )
          {
            // doing 8 of these complex multiplies inside one loop saves a HUGE LOT of processor cycles
              accum[2 * i + 0] +=  fftout[k][2 * i + 0] * fmask[j][2 * i + 0] -
                                   fftout[k][2 * i + 1] * fmask[j][2 * i + 1];
              accum[2 * i + 1] +=  fftout[k][2 * i + 0] * fmask[j][2 * i + 1] +
                                   fftout[k][2 * i + 1] * fmask[j][2 * i + 0]; 
              accum[2 * i + 2] +=  fftout[k][2 * i + 2] * fmask[j][2 * i + 2] -
                                   fftout[k][2 * i + 3] * fmask[j][2 * i + 3];
              accum[2 * i + 3] +=  fftout[k][2 * i + 2] * fmask[j][2 * i + 3] +
                                   fftout[k][2 * i + 3] * fmask[j][2 * i + 2]; 
              accum[2 * i + 4] +=  fftout[k][2 * i + 4] * fmask[j][2 * i + 4] -
                                   fftout[k][2 * i + 5] * fmask[j][2 * i + 5];
              accum[2 * i + 5] +=  fftout[k][2 * i + 4] * fmask[j][2 * i + 5] +
                                   fftout[k][2 * i + 5] * fmask[j][2 * i + 4]; 
              accum[2 * i + 6] +=  fftout[k][2 * i + 6] * fmask[j][2 * i + 6] -
                                   fftout[k][2 * i + 7] * fmask[j][2 * i + 7];
              accum[2 * i + 7] +=  fftout[k][2 * i + 6] * fmask[j][2 * i + 7] +
                                   fftout[k][2 * i + 7] * fmask[j][2 * i + 6]; 
          }
          k = k - 1;
          if(k < 0)
          {
            k = nfor - 1;
          } 

      } // end nfor loop

      buffidx = buffidx + 1;
      if(buffidx >= nfor)
      {
          buffidx = 0;    
      } 
      /**********************************************************************************
          Complex inverse FFT
       **********************************************************************************/
      arm_cfft_f32(iS, accum, 1, 1);

      /**********************************************************************************
          Overlap and save algorithm, which simply means yóu take only half of the buffer
          and discard the other half (which contains unusable time-aliased audio).
          Whether you take the left or the right part is determined by the position
          of the zero-padding in the filter-mask-buffer before doing the FFT of the 
          impulse response coefficients          
       **********************************************************************************/
        for (unsigned i = 0; i < partitionsize; i++)
        {
          //float_buffer_L[i] = accum[partitionsize * 2 + i * 2 + 0];
          //float_buffer_R[i] = accum[partitionsize * 2 + i * 2 + 1];
          float_buffer_L[i] = accum[i * 2 + 0] * audio_gain;
          float_buffer_R[i] = accum[i * 2 + 1] * audio_gain;
        }

     /**********************************************************************************
          Serial print the first output samples in order to check for latency
       **********************************************************************************/
#if defined(LATENCY_TEST)
        if(no_more_latency_test == 1)
        {
          no_more_latency_test = 2;
          for(unsigned i = 0; i < partitionsize; i++)
          {
            if(accum[i * 2 + 0] * audio_gain > 0.1 || accum[i * 2 + 0] * audio_gain < -0.1)
            {
              Serial.print(i + all_samples_counter - 21); Serial.print("   left:   "); Serial.println(accum[i * 2 + 0]);
              Serial.print(i + all_samples_counter - 21); Serial.print("  right:   "); Serial.println(accum[i * 2 + 1]);
            }
          }
        }
#endif

       /**********************************************************************
          CONVERT TO INTEGER AND PLAY AUDIO - Push audio into I2S audio chain
       **********************************************************************/
      for (int i = 0; i < N_BLOCKS; i++)
        {
          sp_L = Q_out_L.getBuffer();    
          sp_R = Q_out_R.getBuffer();
          arm_float_to_q15 (&float_buffer_L[BUFFER_SIZE * i], sp_L, BUFFER_SIZE); 
          arm_float_to_q15 (&float_buffer_R[BUFFER_SIZE * i], sp_R, BUFFER_SIZE);
          Q_out_L.playBuffer(); // play it !  
          Q_out_R.playBuffer(); // play it !
        }

       /**********************************************************************************
          PRINT ROUTINE FOR ELAPSED MICROSECONDS
       **********************************************************************************/
#ifdef DEBUG
      sum = sum + usec;
      idx_t++;
      if (idx_t > 400) {
        mean = sum / idx_t;
        if (mean / 29.00 / N_BLOCKS * SAMPLE_RATE / AUDIO_SAMPLE_RATE_EXACT < 100.0)
        {
          Serial.print("processor load:  ");
          Serial.print (mean / 29.00 / N_BLOCKS * SAMPLE_RATE / AUDIO_SAMPLE_RATE_EXACT);
          Serial.println("%");
        }
        else
        {
          Serial.println("100%");
        }
        Serial.print (mean);
        Serial.print (" microsec for ");
        Serial.print (N_BLOCKS);
        Serial.print ("  stereo blocks    ");
        Serial.print("FFT-length = "); Serial.print(FFT_length);
        Serial.print(";   FIR filter length = "); Serial.println(nc);
        Serial.print("k = "); Serial.println(k);
        Serial.print("buffidx = "); Serial.println(buffidx);
        idx_t = 0;
        sum = 0;
      }
#endif
    } // end of audio process loop
    
      /**********************************************************************************
          Add button check etc. here
       **********************************************************************************/

}

void init_partitioned_filter_masks()
{
    for(unsigned j = 0; j < nfor;j++)
    {
      // fill with zeroes
      for (unsigned i = 0; i < partitionsize * 4; i++)
      {
          maskgen[i] = 0.0;  
      }
      // take part of impulse response and fill into maskgen
      for (unsigned i = 0; i < partitionsize; i++)
      {   
        // THIS IS FOR REAL IMPULSE RESPONSES OR FIR COEFFICIENTS
        // FOR COMPLEX USE THE PART BELOW
          // the position of the impulse response coeffs (right or left aligned)
          // determines the useable part of the audio in the overlap-and-save (left or right part of the iFFT buffer)
          maskgen[i * 2 + partitionsize * 2] = guitar_cabinet_impulse[i + j * partitionsize];  
      }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////      
/*    
 *     THIS COMMENTED OUT PART IS FOR COMPLEX FILTER COEFFS
 *     
 *     // take part of impulse response and fill into maskgen
      for (unsigned i = 0; i < partitionsize * 2; i++)
      {
          // the position of the impulse response coeffs (right or left aligned)
          // determines the useable part of the audio in the overlap-and-save (left or right part of the iFFT buffer)
          maskgen[i + partitionsize * 2] = guitar_cabinet_impulse[i + j * partitionsize * 2];  
      }
      */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////      
      // perform complex FFT on maskgen
      arm_cfft_f32(maskS, maskgen, 0, 1);
      // fill into fmask array
      for (unsigned i = 0; i < partitionsize * 4; i++)
      {
          fmask[j][i] = maskgen[i];  
      }    
    }
}

void setI2SFreq(int freq) {
  // thanks FrankB !
  // PLL between 27*24 = 648MHz und 54*24=1296MHz
  int n1 = 4; //SAI prescaler 4 => (n1*n2) = multiple of 4
  int n2 = 1 + (24000000 * 27) / (freq * 256 * n1);
  double C = ((double)freq * 256 * n1 * n2) / 24000000;
  int c0 = C;
  int c2 = 10000;
  int c1 = C * c2 - (c0 * c2);
  set_audioClock(c0, c1, c2, true);
  CCM_CS1CDR = (CCM_CS1CDR & ~(CCM_CS1CDR_SAI1_CLK_PRED_MASK | CCM_CS1CDR_SAI1_CLK_PODF_MASK))
       | CCM_CS1CDR_SAI1_CLK_PRED(n1-1) // &0x07
       | CCM_CS1CDR_SAI1_CLK_PODF(n2-1); // &0x3f 
//Serial.printf("SetI2SFreq(%d)\n",freq);
}


void flexRamInfo(void)
{ // credit to FrankB, KurtE and defragster !
#if defined(__IMXRT1052__) || defined(__IMXRT1062__)
  int itcm = 0;
  int dtcm = 0;
  int ocram = 0;
  Serial.print("FlexRAM-Banks: [");
  for (int i = 15; i >= 0; i--) {
    switch ((IOMUXC_GPR_GPR17 >> (i * 2)) & 0b11) {
      case 0b00: Serial.print("."); break;
      case 0b01: Serial.print("O"); ocram++; break;
      case 0b10: Serial.print("D"); dtcm++; break;
      case 0b11: Serial.print("I"); itcm++; break;
    }
  }
  Serial.print("] ITCM: ");
  Serial.print(itcm * 32);
  Serial.print(" KB, DTCM: ");
  Serial.print(dtcm * 32);
  Serial.print(" KB, OCRAM: ");
  Serial.print(ocram * 32);
#if defined(__IMXRT1062__)
  Serial.print("(+512)");
#endif
  Serial.println(" KB");
  extern unsigned long _stext;
  extern unsigned long _etext;
  extern unsigned long _sdata;
  extern unsigned long _ebss;
  extern unsigned long _flashimagelen;
  extern unsigned long _heap_start;

  Serial.println("MEM (static usage):");
  Serial.println("RAM1:");

  Serial.print("ITCM = FASTRUN:      ");
  Serial.print((unsigned)&_etext - (unsigned)&_stext);
  Serial.print("   "); Serial.print((float)((unsigned)&_etext - (unsigned)&_stext) / ((float)itcm * 32768.0) * 100.0);
  Serial.print("%  of  "); Serial.print(itcm * 32); Serial.print("kb   ");
  Serial.print("  (");
  Serial.print(itcm * 32768 - ((unsigned)&_etext - (unsigned)&_stext));
  Serial.println(" Bytes free)");
 
  Serial.print("DTCM = Variables:    ");
  Serial.print((unsigned)&_ebss - (unsigned)&_sdata);
  Serial.print("   "); Serial.print((float)((unsigned)&_ebss - (unsigned)&_sdata) / ((float)dtcm * 32768.0) * 100.0);
  Serial.print("%  of  "); Serial.print(dtcm * 32); Serial.print("kb   ");
  Serial.print("  (");
  Serial.print(dtcm * 32768 - ((unsigned)&_ebss - (unsigned)&_sdata));
  Serial.println(" Bytes free)");

  Serial.println("RAM2:");
  Serial.print("OCRAM = DMAMEM:      ");
  Serial.print((unsigned)&_heap_start - 0x20200000);
  Serial.print("   "); Serial.print((float)((unsigned)&_heap_start - 0x20200000) / ((float)512 * 1024.0) * 100.0);
  Serial.print("%  of  "); Serial.print(512); Serial.print("kb");
  Serial.print("     (");
  Serial.print(512 * 1024 - ((unsigned)&_heap_start - 0x20200000));
  Serial.println(" Bytes free)");

  Serial.print("FLASH:               ");
  Serial.print((unsigned)&_flashimagelen);
  Serial.print("   "); Serial.print(((unsigned)&_flashimagelen) / (2048.0 * 1024.0) * 100.0);
  Serial.print("%  of  "); Serial.print(2048); Serial.print("kb");
  Serial.print("    (");
  Serial.print(2048 * 1024 - ((unsigned)&_flashimagelen));
  Serial.println(" Bytes free)");
  
#endif
}
