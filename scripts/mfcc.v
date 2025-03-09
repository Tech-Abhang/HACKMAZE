`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/09/2025 11:07:52 AM
// Design Name: 
// Module Name: mfcc
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

(* KEEP_HIERARCHY = "yes" *)
module mfcc (
    (* DONT_TOUCH = "true" *) 
    (* CLOCK_DEDICATED_ROUTE = "TRUE" *)
    input wire clk,
    input wire [15:0] audio_sample,
    output reg [13*32-1:0] mfcc_coeff
);
    integer i, j;
    // Declare arrays as signed
    reg signed [31:0] mfcc_array [0:12];  
    reg signed [31:0] mel_coeff [0:25];   
    reg signed [31:0] mel_energy [0:25];  
    reg signed [31:0] dct_sum;
    reg signed [31:0] log_value;
    // FFT Module (Xilinx IP Core or custom implementation)
    wire signed [31:0] fft_output;  // Ensure this signal is signed
    fft_processor fft_inst (
        .clk(clk),
        .sample(audio_sample),
        .spectrum(fft_output)
    );
    
    // Register to store FFT output
    reg signed [31:0] fft_output_reg;
    
    // Stage 1: Capture FFT output and start mel filterbank calculation
    always @(posedge clk) begin
        fft_output_reg <= fft_output;
        
        for (i = 0; i < 26; i = i + 1) begin
            mel_energy[i] <= $signed(fft_output_reg) * $signed(mel_coeff[i]);
        end
    end
    
    // Stage 2: Log and DCT calculation
    always @(posedge clk) begin
        for (i = 0; i < 13; i = i + 1) begin
            dct_sum = 0;
            for (j = 0; j < 26; j = j + 1) begin
                // Simple log approximation (replace with actual log computation)
                log_value = mel_energy[j];
                // Apply DCT with explicit signed casting
                dct_sum = dct_sum + ($signed(log_value) * $signed(i + 1) * $signed(j + 1));
            end
            mfcc_array[i] <= dct_sum;
        end
    end
    
    // Stage 3: Pack MFCC coefficients into output
    always @(posedge clk) begin
        for (i = 0; i < 13; i = i + 1) begin
            mfcc_coeff[i*32 +: 32] <= mfcc_array[i];
        end
    end
endmodule