`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/09/2025 12:10:16 PM
// Design Name: 
// Module Name: fetch
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
module audio_file_reader (
  (* DONT_TOUCH = "true" *) 
  (* CLOCK_DEDICATED_ROUTE = "TRUE" *)
  input  wire        clk,            // System clock
  input  wire        rst,            // Synchronous reset (active high)
  // SD card interface signals
  output reg         sd_read_req,    // Request to read a sample from the SD card
  input  wire        sd_data_valid,  // Valid flag from SD card indicating data is ready
  input  wire [7:0]  sd_data,        // 8-bit audio sample from the SD card
  // MFCC module interface signals
  output reg  [7:0]  mfcc_data,      // Audio sample provided to the MFCC module
  output reg         mfcc_data_valid,// Indicates that mfcc_data is valid
  input  wire        mfcc_data_ready // MFCC module ready to accept the sample
);

  // Parameter: Total number of audio samples to read
  parameter NUM_SAMPLES = 1024;

  // FSM state encoding
  localparam IDLE = 2'b00,
             READ = 2'b01,
             SEND = 2'b10,
             DONE = 2'b11;

  reg [1:0] state, next_state;
  reg [15:0] sample_count;  // Counter for the number of samples processed

  // -------------------------------------------------------------------
  // State and Sample Counter Update
  // -------------------------------------------------------------------
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      state         <= IDLE;
      sample_count  <= 16'd0;
    end else begin
      state <= next_state;
      // Increment the sample counter when sending data and MFCC accepts it
      if ((state == SEND) && mfcc_data_ready)
        sample_count <= sample_count + 16'd1;
    end
  end

  // -------------------------------------------------------------------
  // Next State Logic
  // -------------------------------------------------------------------
  always @(*) begin
    next_state = state;  // Default assignment
    case (state)
      IDLE: begin
        // Begin processing immediately after reset
        next_state = READ;
      end
      READ: begin
        // Wait for a valid sample from the SD card
        if (sd_data_valid)
          next_state = SEND;
        else
          next_state = READ;
      end
      SEND: begin
        // Wait for the MFCC module to accept the sample.
        // If we have processed the required number of samples, go to DONE.
        if (mfcc_data_ready) begin
          if (sample_count + 1 >= NUM_SAMPLES)
            next_state = DONE;
          else
            next_state = READ;
        end else begin
          next_state = SEND;
        end
      end
      DONE: begin
        // End state: remain here once all samples have been read
        next_state = DONE;
      end
      default: next_state = IDLE;
    endcase
  end

  // -------------------------------------------------------------------
  // Output Logic
  // -------------------------------------------------------------------
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      sd_read_req     <= 1'b0;
      mfcc_data_valid <= 1'b0;
      mfcc_data       <= 8'd0;
    end else begin
      case (state)
        IDLE: begin
          sd_read_req     <= 1'b0;
          mfcc_data_valid <= 1'b0;
          mfcc_data       <= 8'd0;
        end
        READ: begin
          // Assert read request to prompt the SD card interface to output a sample
          sd_read_req     <= 1'b1;
          mfcc_data_valid <= 1'b0;
          mfcc_data       <= 8'd0;
        end
        SEND: begin
          // De-assert the read request and provide the sample to the MFCC module
          sd_read_req     <= 1'b0;
          mfcc_data_valid <= 1'b1;
          mfcc_data       <= sd_data;
        end
        DONE: begin
          // Once all samples have been processed, disable further read requests
          sd_read_req     <= 1'b0;
          mfcc_data_valid <= 1'b0;
          mfcc_data       <= 8'd0;
        end
        default: begin
          sd_read_req     <= 1'b0;
          mfcc_data_valid <= 1'b0;
          mfcc_data       <= 8'd0;
        end
      endcase
    end
  end

endmodule