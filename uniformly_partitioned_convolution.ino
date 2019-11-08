/* (c) Frank DD4WH 2019-11-02
 *  
 * Real Time PARTITIONED BLOCK CONVOLUTION FILTERING (Stereo)
 * 
 * FIR filtering fun with the Teensy 4.0 with up to 21632 filter taps in each channel
 * 
 * LOW LATENCY is the aim of this approach, it should be no more than 128 samples
 * However, there is additional latency by the codec used and by the overhead of the audio lib
 * 
 * implements a lowpass filter for each channel at a user-defined lowpass cutoff frequency
 * 
 * uses Teensy 4.0 and external ADC / DAC connected on perf board with ground plane
 * but should run also with the PJRC Teensy Audio board version D (compatible with Teensy 4.0)
 * 
 * this sketch does only run with the Teensy 4.0, do not use with Teensy 3.x
 * 
 * My sincere thanks go to Brian Millier, who accurately measured latency in a first version and also found the bug in the system!  
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
 * uses code from wdsp library by Warren Pratt
 * https://github.com/g0orx/wdsp/blob/master/firmin.c
 * 
 ********************************************************************************
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
 ********************************************************************************/

#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <arm_math.h>
#include <arm_const_structs.h> // in the Teensy 4.0 audio library, the ARM CMSIS DSP lib is already a newer version
#include <utility/imxrt_hw.h> // necessary for setting the sample rate, thanks FrankB !

extern "C" uint32_t set_arm_clock(uint32_t frequency);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// USER DEFINES
//uncomment for pass-thru
//#define PASSTHRU_AUDIO

// uncomment for Serial print out of latency
#define LATENCY_TEST

// define your frequency for the lowpass filter
const double FHiCut = 2500.0; // for the (young) audiophile ;-)
const double FLoCut = -FHiCut;
const float32_t audio_gain = 6.5;
// define your sample rate
const double SAMPLE_RATE = 48000;  
const int partitionsize = 128; 

// define the number of FIR taps of your filter
//const int nc = 384;
//const int nc = 128;
//const int nc = 256;
//const int nc = 512; // number of taps for the FIR filter
//const int nc = 1024; // number of taps for the FIR filter
//const int nc = 2048; // number of taps for the FIR filter
//const int nc = 4096; // number of taps for the FIR filter
//const int nc = 8192; // number of taps for the FIR filter
//const int nc = 16384; // number of taps for the FIR filter
const int nc = 28800; // MAXIMUM number of taps for the FIR filter --> memory constraint (99.5%), not processor speed (86%), watch out for stability problems due to low memory !

// RAM1:
// ITCM = FASTRUN:      31440   95.95%  of  32kb     (1328 Bytes free)
// DTCM = Variables:    488128   99.31%  of  480kb     (3392 Bytes free)
// RAM2:
// OCRAM = DMAMEM:      521536   99.48%  of  512kb     (2752 Bytes free)
// FLASH:               45200   2.16%  of  2048kb    (2051952 Bytes free)
// processor load 85.76%

// this is the shift necessary for k, figured out by Brian Millier, thanks a lot !
int kshift = nc / partitionsize / 2 - 1;

// the latency of the filter is meant to be the same regardless of the number of taps for the filter
// partition size of 128 translates to a latency of 128/sample rate, ie. to 2.9msec with 44.1ksps [if the number of taps is a power of two]
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DEBUG
#define FOURPI  (2.0 * TWO_PI)
#define SIXPI   (3.0 * TWO_PI)
#define BUFFER_SIZE 128
int32_t sum;
int idx_t = 0;
int16_t *sp_L;
int16_t *sp_R;
uint8_t FIR_filter_window = 1;
const uint32_t FFT_L = 256; 
uint32_t sample_counter = 0;
uint32_t all_samples_counter = 0;
uint8_t no_more_latency_test = 0;
float32_t mean = 1;
uint8_t first_block = 1; 
const uint32_t FFT_length = FFT_L;
const int nfor = nc / partitionsize; // number of partition blocks --> nfor = nc / partitionsize

const int nc_boundary = (int)(nc / 2);

float32_t DMAMEM cplxcoeffs[nc_boundary]; // this holds the initial complex coefficients for the filter BEFORE partitioning
float32_t maskgen[FFT_L * 2];
float32_t fmask[nfor][FFT_L * 2]; // 
float32_t fftin[FFT_L * 2];
float32_t DMAMEM fftout[nfor][FFT_L * 2]; // 
float32_t accum[FFT_L * 2];

int buffidx = 0;
int k = 0;

const uint32_t N_B = FFT_L / 2 / BUFFER_SIZE;
uint32_t N_BLOCKS = N_B;
float32_t float_buffer_L [BUFFER_SIZE * N_B];  // 
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

AudioConnection          patchCord1(i2s_in, 1, Q_in_L, 0);
AudioConnection          patchCord2(i2s_in, 0, Q_in_R, 0);
AudioConnection          patchCord3(Q_out_L, 0, mixleft, 0);
AudioConnection          patchCord4(Q_out_R, 0, mixright, 0);
AudioConnection          patchCord9(mixleft, 0,  i2s_out, 1);
AudioConnection          patchCord10(mixright, 0, i2s_out, 0);
AudioControlSGTL5000     audio_codec;

void setup() {
  Serial.begin(115200);
  while(!Serial);

  AudioMemory(10); 
  delay(100);

  set_arm_clock(600000000);

  /****************************************************************************************
     Audio Setup
  ****************************************************************************************/
  // Enable the audio shield (if present !). select input, and enable output
  audio_codec.enable();
  audio_codec.inputSelect(AUDIO_INPUT_LINEIN);
  audio_codec.adcHighPassFilterDisable();
  audio_codec.volume(0.5); 

  mixleft.gain (0, 1.0);
  mixright.gain(0, 1.0);

  setI2SFreq(SAMPLE_RATE);

  /****************************************************************************************
     set filter bandwidth
  ****************************************************************************************/
  for(unsigned i = 0; i < nc_boundary; i++)
  {
    cplxcoeffs[i] = 0.0;
  }
  // this routine does all the magic of calculating the FIR coeffs
  fir_bandpass (nc, FLoCut, FHiCut, SAMPLE_RATE, 0, 1, 1.0);

  /****************************************************************************************
     init complex FFTs
  ****************************************************************************************/
  switch (FFT_length)
  {
    case 256:
      S = &arm_cfft_sR_f32_len256;
      iS = &arm_cfft_sR_f32_len256;
      maskS = &arm_cfft_sR_f32_len256;
      break;
  }

  /****************************************************************************************
     initialise variables in DMAMEM
  ****************************************************************************************/
  for(unsigned jj = 0; jj < nfor; jj++)
  {
    for(unsigned ii = 0; ii < FFT_L * 2; ii++)
    {
      fftout[jj][ii] = 0.0;
    }
  }

  /****************************************************************************************
     Calculate the FFT of the FIR filter coefficients once to produce the FIR filter mask
  ****************************************************************************************/
    init_partitioned_filter_masks();
 
  /****************************************************************************************
     display info on memory usage and convolution stuff
  ****************************************************************************************/
    flexRamInfo();
    Serial.println();
    Serial.print("Number of blocks:   "); Serial.println(nfor);
    Serial.println();    
    Serial.print("number of k shift:   "); Serial.println(kshift);
    Serial.println();    
    Serial.print("FIR filter length = "); Serial.println(nc);
    Serial.println();    
    
  /****************************************************************************************
     begin to queue the audio from the audio library
  ****************************************************************************************/
  delay(100);
  Q_in_L.begin();
  Q_in_R.begin();

} // END OF SETUP

void loop() {
  elapsedMicros usec = 0;
  // are there at least N_BLOCKS buffers in each channel available ?
    if (Q_in_L.available() > N_BLOCKS && Q_in_R.available() > N_BLOCKS)
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
          Fast convolution filtering == FIR filter in the frequency domain
       **********************************************************************************/
      //  basis for this was Lyons, R. (2011): Understanding Digital Processing.
      //  "Fast FIR Filtering using the FFT", pages 688 - 694
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

#if defined(LATENCY_TEST)
        if(all_samples_counter > SAMPLE_RATE && !no_more_latency_test)
        {
        // latency test
        fftin[42] = 1.0; fftin[44] = -1.0;
        fftin[43] = 1.0; fftin[45] = -1.0;
        no_more_latency_test = 1;
        }
      if(!no_more_latency_test) all_samples_counter += partitionsize; 
#endif           

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
          however: modified k --> originally was without kshift, which lead to high latency
       **********************************************************************************/
      k = buffidx;

      for(unsigned i = 0; i < partitionsize * 4; i++)
      {
          accum[i] = 0.0;
      }
      
      for(unsigned j = 0; j < nfor; j++)
      { 
          for(unsigned i = 0; i < 2 * partitionsize; i++)
          {
/*              accum[2 * i + 0] += fftout[(k + kshift) % nfor][2 * i + 0] * fmask[j][2 * i + 0] -
                                  fftout[(k + kshift) % nfor][2 * i + 1] * fmask[j][2 * i + 1];
              
              accum[2 * i + 1] += fftout[(k + kshift) % nfor][2 * i + 0] * fmask[j][2 * i + 1] +
                                  fftout[(k + kshift) % nfor][2 * i + 1] * fmask[j][2 * i + 0]; 
*/
              accum[2 * i + 0] += fftout[k][2 * i + 0] * fmask[j][2 * i + 0] -
                                  fftout[k][2 * i + 1] * fmask[j][2 * i + 1];
              
              accum[2 * i + 1] += fftout[k][2 * i + 0] * fmask[j][2 * i + 1] +
                                  fftout[k][2 * i + 1] * fmask[j][2 * i + 0]; 
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
          Overlap and save algorithm, which simply means yóu take only the right part of the buffer and discard the left part
       **********************************************************************************/
      // I reversed the reversion of this, so this is history: "somehow it seems it is reversed: I discard the right part and take the left part . . ."
      // see in function init_filter_masks -->  
        for (unsigned i = 0; i < partitionsize; i++)
        { // the right part of the iFFT output is unaliased audio
          float_buffer_L[i] = accum[partitionsize * 2 + i * 2 + 0] * audio_gain;
          float_buffer_R[i] = accum[partitionsize * 2 + i * 2 + 1] * audio_gain;
          //float_buffer_L[i] = accum[i * 2 + 0] * audio_gain;
          //float_buffer_R[i] = accum[i * 2 + 1] * audio_gain;
        }

      /**********************************************************************************
          Serial print the first output samples in order to check for latency
       **********************************************************************************/
#if defined(LATENCY_TEST)
        if(sample_counter < nc * 4 + SAMPLE_RATE)
        {
          for(unsigned i = 0; i < partitionsize; i++)
          {
            if(accum[i * 2 + 0] > 0.1 || accum[i * 2 + 0] < -0.1)
            {
              Serial.print(sample_counter + i - all_samples_counter - 21); Serial.print("   left:   "); Serial.println(accum[i * 2 + 0]);
              Serial.print(sample_counter + i - all_samples_counter - 21); Serial.print("  right:   "); Serial.println(accum[i * 2 + 1]);
            }
          }
          sample_counter += partitionsize;
        }
        else
        {
          sample_counter = nc * 4;
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
      if (idx_t > 500) {
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
} // end main loop

void init_partitioned_filter_masks()
{
///////////////////////////////////////////////////////////////
    // first half of impulse response up to nfor/2
    for(int j = 0; j < nfor / 2; j++)
    {
      // fill with zeroes
      for (int i = 0; i < partitionsize * 4; i++)
      {
          maskgen[i] = 0.0;  
      }
      // take part of impulse response and fill into maskgen
      for (int i = 0; i < partitionsize; i++)
      { 
        // this makes the unaliased output appear in the right part of the iFFT buffer
          maskgen[i * 2 + 0] = cplxcoeffs[i + j * partitionsize];
          maskgen[i * 2 + 1] = 0.0;  
//          Serial.print("cplxcoeffs[] "); Serial.println(i + j * partitionsize);
      }
      // perform complex FFT on maskgen
      arm_cfft_f32(maskS, maskgen, 0, 1);
      // fill into fmask array
      for (int i = 0; i < partitionsize * 4; i++)
      {
          fmask[j][i] = maskgen[i];  
      }    
    }

    // second half of impulse response (symmetric copy of the first half)
    for(int j = 0; j < nfor / 2; j++)
    {
      // fill with zeroes
      for (int i = 0; i < partitionsize * 4; i++)
      {
          maskgen[i] = 0.0;  
      }
      // take part of impulse response and fill into maskgen
      for (int i = 0; i < partitionsize; i++)
      { 
        // this makes the unaliased output appear in the right part of the iFFT buffer
        // we have to take the same symmetric coeffs, but now start from behind
          maskgen[i * 2 + 0] = cplxcoeffs[((int)(nc / 2) - 1) - i - (j * partitionsize)];
          maskgen[i * 2 + 1] = 0.0;  
//          Serial.print("cplxcoeffs[] "); Serial.println(((int)(nc / 2) - 1) - i - (j * partitionsize));
      }
      // perform complex FFT on maskgen
      arm_cfft_f32(maskS, maskgen, 0, 1);
      // fill into fmask array
      for (int i = 0; i < partitionsize * 4; i++)
      {
        // important to add nfor/2 to index !
          fmask[j + nfor / 2][i] = maskgen[i];  
      }    
    }
}

// taken from wdsp library by Warren Pratt
void fir_bandpass (int N, double f_low, double f_high, double samplerate, int wintype, int rtype, double scale)
{
  double ft = (f_high - f_low) / (2.0 * samplerate);
  double ft_rad = TWO_PI * ft;
  double w_osc = PI * (f_high + f_low) / samplerate;
  int i, j;
  double m = 0.5 * (double)(N - 1);
  double delta = PI / m;
  double cosphi;
  double posi, posj;
  double sinc, window, coef;

  for (i = (N + 1) / 2, j = N / 2 - 1; i < N; i++, j--)
  {
    posi = (double)i - m;
    posj = (double)j - m;
    sinc = sin (ft_rad * posi) / (PI * posi);
    switch (wintype)
    {
    case 0: // Blackman-Harris 4-term
      cosphi = cos (delta * i);
      window  =             + 0.21747
          + cosphi *  ( - 0.45325
          + cosphi *  ( + 0.28256
          + cosphi *  ( - 0.04672 )));
      break;
    case 1: // Blackman-Harris 7-term
      cosphi = cos (delta * i);
      window  =       + 6.3964424114390378e-02
          + cosphi *  ( - 2.3993864599352804e-01
          + cosphi *  ( + 3.5015956323820469e-01
          + cosphi *  ( - 2.4774111897080783e-01
          + cosphi *  ( + 8.5438256055858031e-02
          + cosphi *  ( - 1.2320203369293225e-02
          + cosphi *  ( + 4.3778825791773474e-04 ))))));
      break;
    }
    coef = scale * sinc * window;
    switch (rtype)
    {
    case 1:
      cplxcoeffs[j] = + coef * cos (posj * w_osc);
      Serial.print(j); Serial.print("  =  "); Serial.println(cplxcoeffs[j] * 1000000);
      break;
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
