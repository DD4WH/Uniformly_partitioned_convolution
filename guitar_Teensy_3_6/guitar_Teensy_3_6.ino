/*
 * (c) DD4WH 15/11/2019
 * 
 * Real Time PARTITIONED BLOCK CONVOLUTION FILTERING (STEREO)
 * 
 * thanks a lot to Brian Millier and Warren Pratt for your help!
 * 
 * using a guitar cabinet impulse response with up to about 4096 coefficients per channel
 * 
 * uses Teensy 3.6 and the Teensy audio shield
 * 
 * inspired by and uses code from wdsp library by Warren Pratt
 * https://github.com/g0orx/wdsp/blob/master/firmin.c
 * 
*********************************************************************************

   GNU GPL LICENSE v3

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>

 *********************************************************************************
 */

// processor load Teensy 3.6 with 180MHz
// 512 taps 20.89%
// 1024 taps 26.30%
// 2048 taps 36.47 %
// 4096 taps 54.08%
// 7552 taps 84.90% 98% RAM USE !!!!

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <arm_math.h>
#include <arm_const_structs.h> 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// USER DEFINES

// Choose only one of these impulse responses !

//#define IR1  // 512 taps // MG impulse response from bmillier github @44.1ksps
//#define IR2  // 4096 taps // impulse response @44.1ksps
//#define IR3  // 7552 taps // impulse response @44.1ksps
//#define IR4    // 17920 taps // impulse response 400ms @44.1ksps
//#define IR5    // 21632 taps // impulse response 490ms @44.1ksps
#define IR6 // 5760 taps
//#define LPMINPHASE512 // 512 taps minimum phase 2.7kHz lowpass filter
//#define LPMINPHASE1024 // 1024 taps minimum phase 2.7kHz lowpass filter
//#define LPMINPHASE2048PASSTHRU // 2048 taps minimum phase 19.0kHz lowpass filter
//#define LPMINPHASE4096 // 4096 taps minimum phase 2.7kHz lowpass filter
const float32_t PROGMEM audio_gain = 0.2; // has to be adjusted from 1.0 to 10.0 depending on the filter gain / impulse resonse gain
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
const int nc = 5760; // number of taps for the FIR filter
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

//extern "C" uint32_t set_arm_clock(uint32_t frequency);

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
//float32_t maskgen[FFT_L * 2];
float32_t fmask[nfor][FFT_L * 2]; // 
//float32_t fftin[FFT_L * 2];
float32_t accum[FFT_L * 2];
float  fftout[nfor][FFT_L * 2]; // 

int buffidx = 0;
int k = 0;
//int idxmask = nfor - 1;

uint32_t all_samples_counter = 0;
uint8_t no_more_latency_test = 0;

const uint32_t N_B = FFT_L / 2 / BUFFER_SIZE;
uint32_t N_BLOCKS = N_B;
float32_t float_buffer_L [BUFFER_SIZE * N_B];  
float32_t float_buffer_R [BUFFER_SIZE * N_B]; 
float32_t last_sample_buffer_L [BUFFER_SIZE * N_B];  
float32_t last_sample_buffer_R [BUFFER_SIZE * N_B]; 
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
AudioControlSGTL5000     codec;

// I2S audio is sent to the queue, the software treats the audio
AudioConnection          patchCord1(i2s_in, 0, Q_in_L, 0);
AudioConnection          patchCord2(i2s_in, 1, Q_in_R, 0);

// the audio treated by the software is sent back to the queue and to the mixers
AudioConnection          patchCord3(Q_out_L, 0, mixleft, 0);
AudioConnection          patchCord4(Q_out_R, 0, mixright, 0);

// the mixers output is sent to the I2S output channels
AudioConnection          patchCord9(mixleft, 0,  i2s_out, 1);
AudioConnection          patchCord10(mixright, 0, i2s_out, 0);


void setup() {
  Serial.begin(115200);
  if(!Serial) delay (2000);

  AudioMemory(10); 
  delay(100);

  /****************************************************************************************
     Audio Setup
  ****************************************************************************************/
  // Enable the audio shield. select input. and enable output
  codec.enable();
  codec.adcHighPassFilterDisable(); // can help in certain situations to mitigate high alias noise frequencies, in my case it solved severe noise problems!
  codec.inputSelect(AUDIO_INPUT_LINEIN); // AUDIO_INPUT_MIC
  //codec.micGain(25); // 0 to 63 dB
  codec.lineInLevel(7);
//  0: 3.12 Volts p-p
//  1: 2.63 Volts p-p
//  2: 2.22 Volts p-p
//  3: 1.87 Volts p-p
//  4: 1.58 Volts p-p
//  5: 1.33 Volts p-p
//  6: 1.11 Volts p-p
//  7: 0.94 Volts p-p
//  8: 0.79 Volts p-p
//  9: 0.67 Volts p-p
// 10: 0.56 Volts p-p
// 11: 0.48 Volts p-p
// 12: 0.40 Volts p-p
// 13: 0.34 Volts p-p
// 14: 0.29 Volts p-p
// 15: 0.24 Volts p-p
  codec.audioPostProcessorEnable(); // enables the DAP chain of the codec post audio processing before the headphone out
  //  codec.eqSelect (2); // 2-band Tone Control
  //  codec.eqBands (bass, treble); // (float bass, float treble) in % -100 to +100

  codec.eqSelect (3); // five-band-graphic equalizer
//  codec.eqBands (bass, midbass, mid, midtreble, treble); // (float bass, etc.) in % -100 to +100
  codec.eqBands   (+0.1, -0.2, 0.0, +0.2, 0.0); // (float bass, etc.) in % -100 to +100
  //  codec.enhanceBassEnable();
  //codec.dacVolumeRamp();
  codec.volume(0.5); //

  mixleft.gain(0, 1.0);
  mixright.gain(0, 1.0);

  /****************************************************************************************
     properly initialise variables
  ****************************************************************************************/

  for(unsigned jj = 0; jj < nfor; jj++)
  {
    for(unsigned ii = 0; ii < FFT_L * 2; ii++)
    {
      fftout[jj][ii] = 0.1;
    }
  }

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
          accum[i] = 0.0;
        }
        first_block = 0;
      }
      else
      {  // HERE IT STARTS for all other instances
        // fill FFT_buffer with last events audio samples
        for (unsigned i = 0; i < partitionsize; i++)
        {
          accum[i * 2] = last_sample_buffer_L[i]; // real
          accum[i * 2 + 1] = last_sample_buffer_R[i]; // imaginary
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
        accum[FFT_length + i * 2] = float_buffer_L[i]; // real
        accum[FFT_length + i * 2 + 1] = float_buffer_R[i]; // imaginary
      }

#if defined(LATENCY_TEST)
        if(msec > 2000 && !no_more_latency_test)
        {
        // latency test
        accum[42] = 10.0; accum[44] = -10.0;
        accum[43] = 10.0; accum[45] = -10.0;
        no_more_latency_test = 1;
        }
        if(no_more_latency_test == 1) all_samples_counter += partitionsize; 
#endif       
      /**********************************************************************************
          Complex Forward FFT
       **********************************************************************************/
      // calculation is performed in-place the FFT_buffer [re, im, re, im, re, im . . .]
      arm_cfft_f32(S, accum, 0, 1);
      for(unsigned i = 0; i < partitionsize * 4; i++)
      {
          fftout[buffidx][i] = accum[i];
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
            // doing 8 of these MAC operations inside one loop saves a lot of processor cycles
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
//        Serial.print("k = "); Serial.println(k);
//        Serial.print("buffidx = "); Serial.println(buffidx);
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
          accum[i] = 0.0;  
      }
      // take part of impulse response and fill into maskgen
      for (unsigned i = 0; i < partitionsize; i++)
      {   
        // THIS IS FOR REAL IMPULSE RESPONSES OR FIR COEFFICIENTS
          // the position of the impulse response coeffs (right or left aligned)
          // determines the useable part of the audio in the overlap-and-save (left or right part of the iFFT buffer)
          accum[i * 2 + partitionsize * 2] = guitar_cabinet_impulse[i + j * partitionsize];  
      }
      // perform complex FFT on maskgen
      arm_cfft_f32(maskS, accum, 0, 1);
      // fill into fmask array
      for (unsigned i = 0; i < partitionsize * 4; i++)
      {
          fmask[j][i] = accum[i];  
      }    
    }
}
