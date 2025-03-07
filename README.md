# HACKMAZE
# Hear - it - Right
Design and implement a speech-to-text system on an FPGA that can process audio input, detect specific keywords, and display the recognized words as text. The system should be optimized for real-time performance while efficiently utilizing FPGA resources.
# Key Objectives:
1. Speech-to-Text Conversion: Process voice input and convert it into textual representation.
2. Keyword Spotting: Accurately recognize specific keywords from the speech input. 
Keywords to be detected are : "up", "down", "right", "left", "stop", "go", "Hack Maze", "Triple IT", "Dharwad", "Hubli"
3. FPGA Implementation: Optimize the design for execution on an FPGA.
4.  Resource Efficiency: Ensure minimal usage of LUTs, Flip-Flops, DSPs, and DRAM for better performance.
5. Low Latency: Achieve fast response times for keyword detection. (Inference time)
6.  Scalability: Should be able to detect above keywords in various backgrounds.
7. Power Optimization: Reduce FPGA power consumption for efficient operation.
 8. Gender Detection: Classify the speaker as Male/Female based on voice features.
# Bonus Objectives:
1. User Interface (UI): Implement an interactive interface to display recognized words.

# Expected Outcomes:
1. A functional speech processing system implemented on an FPGA.
2. High-accuracy keyword detection from voice input.
3. Optimized FPGA design using minimal hardware resources.
4. Low-latency real-time speech recognition.
5. Scalable system that can support additional keywords.
6. Demonstration of UI (if implemented).
7. Potential gender classification .

# Evaluation Criteria:
1. Accuracy – Performance of keyword recognition and speech-to-text conversion.
2. Hardware Optimization – Efficient use of FPGA resources (LUTs, FFs, DSPs, BRAM).
3. Power Consumption – How efficiently the system runs on FPGA.
4. Latency – Speed of speech processing and response time. [inference time]

# Deliverables:
1. Presentation (PPT/Report) with a block diagram, approach flow, and system design details.
2. Demonstration Video showcasing real-time speech-to-text functionality.
3. Codebase (Preferably hosted on GitHub).
# Final evaluation criteria: [ includes even your check point submissions]
1. for right detection of following words - up, down, left, right, stop, go would get 18 marks in total. And for right detection of words - Hack Maze, triple IT, Hubli, Dharwad would get 32 marks in total
2. Optimised hardware - 20 marks
3. Power consumption during synthesis - 5 marks
4. Inference time - 15 marks
5. Clarity provied/ demonstrations/ presentation - 10 marks
