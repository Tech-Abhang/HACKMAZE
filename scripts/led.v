`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/09/2025 11:57:58 AM
// Design Name: 
// Module Name: led
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
module keyword_led_controller (
    (* DONT_TOUCH = "true" *) 
    (* CLOCK_DEDICATED_ROUTE = "TRUE" *)
    input  wire        clk,          // System clock
    input  wire        rst,          // Synchronous reset (active high)
    input  wire [3:0]  keyword_index, // Keyword index from Viterbi module
    output reg         led           // LED output to turn on the light
);
// In this simple example, we assume that any non-zero keyword_index indicates a valid keyword.
// When a valid keyword is detected, the LED is turned on.
// When keyword_index is 0, the LED remains off.
always @(posedge clk or posedge rst) begin
    if (rst)
        led <= 1'b0;
    else begin
        if (keyword_index != 4'd0)
            led <= 1'b1;
        else
            led <= 1'b0;
    end
end
endmodule