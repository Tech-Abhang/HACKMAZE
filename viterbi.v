`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/09/2025 09:47:58 AM
// Design Name: 
// Module Name: viterbi
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


//module viterbi (
//    input wire clk,               // Clock signal
//    input wire reset,             // Active-high reset
//    input wire [13*32-1:0] mfcc_coeff, // 13 MFCC features
//    output reg [3:0] keyword_index // Detected keyword index (0-9)
//);
//   integer i;
//   reg [31:0] mfcc_array [0:12];
   
//   reg [31:0] mel_coeff [0:25];
   
//   always @(posedge clk) begin
//        // Example: pack data into mfcc_coeff
//        for (i = 0; i < 13; i = i + 1) begin
//            mfcc_coeff[i*32 +: 32] <= mfcc_array[i];
//        end
//    end
//// Parameters
//parameter N_STATES = 5;          // Number of states per HMM
//parameter N_KEYWORDS = 10;       // Number of keywords
//parameter THRESHOLD = 1000;      // Detection threshold

//// File paths for memory initialization
//parameter TRANS_FILE = "C:\Users\chinm\project_4\project_4.srcs\sources_1\new\transition_matrix.mem";
//parameter MEAN_FILE = "C:\Users\chinm\project_4\project_4.srcs\sources_1\new\emission_mean.mem";
//parameter VAR_FILE = "C:\Users\chinm\project_4\project_4.srcs\sources_1\new\emission_var.mem";

//// HMM Parameters (Precomputed and stored in Block RAM)
//reg [31:0] transition_matrix [0:N_KEYWORDS-1][0:N_STATES-1][0:N_STATES-1]; // Transition matrices
//reg [31:0] emission_mean [0:N_KEYWORDS-1][0:N_STATES-1][0:12];             // Emission means
//reg [31:0] emission_var [0:N_KEYWORDS-1][0:N_STATES-1][0:12];              // Emission variances

//// Initialize HMM Parameters using $readmemh
//initial begin
//    // Load transition matrices
//    $readmemh(TRANS_FILE, transition_matrix);
    
//    // Load emission means
//    $readmemh(MEAN_FILE, emission_mean);
    
//    // Load emission variances
//    $readmemh(VAR_FILE, emission_var);
//end

//// Viterbi Variables
//reg [31:0] viterbi_table [0:N_KEYWORDS-1][0:N_STATES-1]; // Probabilities for each HMM and state
//reg [31:0] max_prob;
//reg [31:0] final_prob [0:N_KEYWORDS-1]; // Final probabilities for each HMM
//integer state, prev_state, keyword;
//reg [31:0] log_prob = 0;
//reg [31:0] diff = mfcc_coeff[i] - emission_mean[keyword][state][i];
//reg [31:0] max_final_prob = 0;
//reg [3:0] detected_keyword = 0;

//always @(posedge clk) begin
//    if (reset) begin
//        // Initialize Viterbi tables and final probabilities
//        for (keyword = 0; keyword < N_KEYWORDS; keyword = keyword + 1) begin
//            for (state = 0; state < N_STATES; state = state + 1) begin
//                viterbi_table[keyword][state] <= 0;
//            end
//            final_prob[keyword] <= 0;
//        end
//        keyword_index <= 0;
//    end else begin
//        // Compute Viterbi probabilities for each keyword
//        for (keyword = 0; keyword < N_KEYWORDS; keyword = keyword + 1) begin
//            for (state = 0; state < N_STATES; state = state + 1) begin
//                // Compute emission probability (Gaussian PDF approximation)
                
//                for (i = 0; i < 13; i = i + 1) begin
//                    // log_prob += (mfcc_coeff[i] - emission_mean[keyword][state][i])^2 / emission_var[keyword][state][i]
                    
//                    log_prob = log_prob + ((diff * diff) / emission_var[keyword][state][i]);
//                end

//                // Update Viterbi table with max(prev_state + transition)
//                max_prob = 0;
//                for (prev_state = 0; prev_state < N_STATES; prev_state = prev_state + 1) begin
//                    if (viterbi_table[keyword][prev_state] + transition_matrix[keyword][prev_state][state] > max_prob)
//                        max_prob <= viterbi_table[keyword][prev_state] + transition_matrix[keyword][prev_state][state];
//                end
//                viterbi_table[keyword][state] <= max_prob + log_prob;
//            end

//            // Store final probability for the keyword
//            final_prob[keyword] <= viterbi_table[keyword][N_STATES-1];
//        end

//        // Find the keyword with the highest final probability
        
//        for (keyword = 0; keyword < N_KEYWORDS; keyword = keyword + 1) begin
//            if (final_prob[keyword] > max_final_prob && final_prob[keyword] > THRESHOLD) begin
//                max_final_prob <= final_prob[keyword];
//                detected_keyword <= keyword;
//            end
//        end
//        keyword_index <= detected_keyword;
//    end
//end

//endmodule

(* KEEP_HIERARCHY = "yes" *)
module viterbi (
    (* DONT_TOUCH = "true" *) 
    (* CLOCK_DEDICATED_ROUTE = "TRUE" *)
    input wire clk,               // Clock signal
    input wire reset,             // Active-high reset
    input wire [13*32-1:0] mfcc_coeff, // 13 MFCC features - this is an INPUT
    output reg [3:0] keyword_index // Detected keyword index (0-9)
);
   integer i;
   reg [31:0] mfcc_array [0:12];
   
   reg [31:0] mel_coeff [0:25];
   
   // Instead of trying to assign to mfcc_coeff, we should extract values FROM it
   always @(posedge clk) begin
        // Extract MFCC coefficients from input into our array
        for (i = 0; i < 13; i = i + 1) begin
            mfcc_array[i] <= mfcc_coeff[i*32 +: 32];
        end
    end

// Parameters
parameter N_STATES = 5;          // Number of states per HMM
parameter N_KEYWORDS = 10;       // Number of keywords
parameter THRESHOLD = 1000;      // Detection threshold

// File paths for memory initialization
parameter TRANS_FILE = "C:/Users/chinm/project_4/project_4.srcs/sources_1/new/transition_matrix.mem";
parameter MEAN_FILE = "C:/Users/chinm/project_4/project_4.srcs/sources_1/new/emission_mean.mem";
parameter VAR_FILE = "C:/Users/chinm/project_4/project_4.srcs/sources_1/new/emission_var.mem";

// HMM Parameters (Precomputed and stored in Block RAM)
reg [31:0] transition_matrix [0:N_KEYWORDS-1][0:N_STATES-1][0:N_STATES-1]; // Transition matrices
reg [31:0] emission_mean [0:N_KEYWORDS-1][0:N_STATES-1][0:12];             // Emission means
reg [31:0] emission_var [0:N_KEYWORDS-1][0:N_STATES-1][0:12];              // Emission variances

// Initialize HMM Parameters using $readmemh
initial begin
    // Load transition matrices
    $readmemh(TRANS_FILE, transition_matrix);
    
    // Load emission means
    $readmemh(MEAN_FILE, emission_mean);
    
    // Load emission variances
    $readmemh(VAR_FILE, emission_var);
end

// Viterbi Variables
reg [31:0] viterbi_table [0:N_KEYWORDS-1][0:N_STATES-1]; // Probabilities for each HMM and state
reg [31:0] max_prob;
reg [31:0] final_prob [0:N_KEYWORDS-1]; // Final probabilities for each HMM
integer state, prev_state, keyword;
reg [31:0] log_prob;
reg [31:0] diff;
reg [31:0] max_final_prob;
reg [3:0] detected_keyword;

always @(posedge clk) begin
    if (reset) begin
        // Initialize Viterbi tables and final probabilities
        for (keyword = 0; keyword < N_KEYWORDS; keyword = keyword + 1) begin
            for (state = 0; state < N_STATES; state = state + 1) begin
                viterbi_table[keyword][state] <= 0;
            end
            final_prob[keyword] <= 0;
        end
        keyword_index <= 0;
        max_final_prob <= 0;
        detected_keyword <= 0;
    end else begin
        // Compute Viterbi probabilities for each keyword
        for (keyword = 0; keyword < N_KEYWORDS; keyword = keyword + 1) begin
            for (state = 0; state < N_STATES; state = state + 1) begin
                // Compute emission probability (Gaussian PDF approximation)
                log_prob = 0; // Reset log_prob for each state
                
                for (i = 0; i < 13; i = i + 1) begin
                    diff = mfcc_array[i] - emission_mean[keyword][state][i];
                    log_prob = log_prob + ((diff * diff) / emission_var[keyword][state][i]);
                end

                // Update Viterbi table with max(prev_state + transition)
                max_prob = 0;
                for (prev_state = 0; prev_state < N_STATES; prev_state = prev_state + 1) begin
                    if (viterbi_table[keyword][prev_state] + transition_matrix[keyword][prev_state][state] > max_prob)
                        max_prob = viterbi_table[keyword][prev_state] + transition_matrix[keyword][prev_state][state];
                end
                viterbi_table[keyword][state] <= max_prob + log_prob;
            end

            // Store final probability for the keyword
            final_prob[keyword] <= viterbi_table[keyword][N_STATES-1];
        end

        // Find the keyword with the highest final probability
        max_final_prob = 0; // Reset max probability
        detected_keyword = keyword_index; // Keep current detection if no new one is found
        
        for (keyword = 0; keyword < N_KEYWORDS; keyword = keyword + 1) begin
            if (final_prob[keyword] > max_final_prob && final_prob[keyword] > THRESHOLD) begin
                max_final_prob = final_prob[keyword];
                detected_keyword = keyword;
            end
        end
        keyword_index <= detected_keyword;
    end
end

endmodule