// from https://github.com/OpenTimer/OpenTimer/blob/master/benchmark/simple/simple.v

module simple (
inp1,
inp2,
tau2015_clk,
out
);

// Start PIs
input inp1;
input inp2;
input tau2015_clk;

// Start POs
output out;

// Start wires
wire n1;
wire n2;
wire n3;
wire n4;
wire inp1;
wire inp2;
wire tau2015_clk;
wire out;

// Start cells
NAND2_X1 u1 ( .a(inp1), .b(inp2), .o(n1) );
DFF_X80 f1 ( .d(n2), .ck(tau2015_clk), .q(n3) );
NAND2_X1 u2 ( .a(1'b1), .b(n3), .o(n4) );
NOR2_X1 u3 ( .a(n4), .b(1'b0), .o(out) );
NOR2_X1 u4 ( .a(n3), .b(n1), .o(n2) );

endmodule
