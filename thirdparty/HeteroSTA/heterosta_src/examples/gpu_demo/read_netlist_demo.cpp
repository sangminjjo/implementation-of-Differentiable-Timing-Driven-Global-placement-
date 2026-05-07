#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <heterosta.h>

// C-style callback function to route Rust logger messages to the C++ standard streams.
extern "C" void cpp_log_callback(uint8_t level, const char* message) {
    const char* level_str;
    switch (level) {
        case 1: level_str = "ERROR"; break;
        case 2: level_str = "WARN "; break;
        case 3: level_str = "INFO "; break;
        case 4: level_str = "DEBUG"; break;
        case 5: level_str = "TRACE"; break;
        default: level_str = "UNKNW"; break;
    }
    
    // Route critical messages to stderr, others to stdout.
    if (level <= 2) {
        std::cerr << "[" << level_str << "] " << message << std::endl;
    } else {
        std::cout << "[" << level_str << "] " << message << std::endl;
    }
}

// A hardcoded fallback license..
const char* hardcode_lic = nullptr;

int main(int argc, char* argv[]) {
    // --- Configuration ---
    const char* netlist_verilog = "../common/data/simple.v";
    const char* liberty_min = "../common/data/simple_Early.lib";
    const char* liberty_max = "../common/data/simple_Late.lib";
    const char* spef_file = "../common/data/simple.spef";
    const char* sdc_file = "../common/data/simple.sdc";
    const char* sdf_out_path = "results/read_netlist_demo/output.sdf";
    const char* rpt_out_path = "results/read_netlist_demo/output.rpt";
    bool use_cuda = true;
    
    // 1. Initialize the logger and license.
    // This sets up the logging system and validates the license key.
    const char* lic = std::getenv("HeteroSTA_Lic");
    if (lic == nullptr) {
        if (hardcode_lic == nullptr) {
            std::cerr << "[FATAL ERROR] License not found." << std::endl;
            std::cerr << "               Please either set the 'HeteroSTA_Lic' environment variable" << std::endl;
            std::cerr << "               or provide a hardcoded license in the source code." << std::endl;
            return 1;
        }
        lic = hardcode_lic;
        std::cout << "[INFO] 'HeteroSTA_Lic' environment variable not found. Using hardcoded license." << std::endl;
    } else {
        std::cout << "[INFO] Successfully loaded license from 'HeteroSTA_Lic' environment variable." << std::endl;
    }
    heterosta_init_logger(cpp_log_callback);
    heterosta_init_license(lic);

    // 2. Create a new STA instance (STAHoldings).
    // This object will hold all the data and state for the timing analysis.
    STAHoldings* sta = heterosta_new();
    if (!sta) {
        std::cerr << "Error: Failed to create STAHoldings instance." << std::endl;
        return 1;
    }
    // Set the delay calculator model. Elmore is a common choice.
    heterosta_set_delay_calculator_elmore(sta);

    // --- Main STA Flow ---
    
    // 3. Load design files into the STA instance.
    // Load the minimum (Early) and maximum (Late) Liberty timing libraries.
    if (!heterosta_read_liberty(sta, EARLY, liberty_min) || !heterosta_read_liberty(sta, LATE, liberty_max)) {
        std::cerr << "Error: Failed on Reading Liberty files." << std::endl;
        heterosta_free(sta);
        return 1;
    }
    
    // Directly read the netlist from a Verilog file.
    // This is the high-level, simplified approach.
    if (!heterosta_read_netlist(sta, netlist_verilog, nullptr)) {
        std::cerr << "Error: Failed on Reading Netlist." << std::endl;
        heterosta_free(sta);
        return 1;
    }

    // Read the parasitic information from a SPEF file.
    if (!heterosta_read_spef(sta, spef_file)) {
        std::cerr << "Error: Failed on Reading SPEF." << std::endl;
        heterosta_free(sta);
        return 1;
    }

    // 4. Build the timing graph.
    // These steps are necessary to construct the internal graph representation for analysis.
    heterosta_flatten_all(sta);
    heterosta_build_graph(sta);
    heterosta_zero_slew(sta);
    
    // 5. Read timing constraints from an SDC file and run STA.
    if (!heterosta_read_sdc(sta, sdc_file)) {
        std::cerr << "Error: Failed on Reading SDC." << std::endl;
        heterosta_free(sta);
        return 1;
    }
    
    // Perform the core timing analysis calculations.
    heterosta_update_delay(sta, use_cuda);
    heterosta_update_arrivals(sta, use_cuda);

    // 6. Report the timing analysis results.
    float wns = 0.0, tns = 0.0;
    if (heterosta_report_wns_tns_max(sta, &wns, &tns, use_cuda)) {
         std::cout << "\n--- Timing Results ---\n"
                   << "WNS (setup): " << wns << "\n"
                   << "TNS (setup): " << tns << std::endl;
    } else {
        std::cerr << "Warning: Could not report WNS/TNS." << std::endl;
    }
    
    // Optionally, write out the calculated delays to an SDF file.
    if (sdf_out_path) {
        if (!heterosta_report_delay_sdf(sta, sdf_out_path)) {
            std::cerr << "Warning: Failed to write SDF file." << std::endl;
        }
    }

    // Optionally, dump the calculated critical paths to an text file.
    if (rpt_out_path) {
        heterosta_dump_paths_max_to_file(sta, 100, 100, rpt_out_path, use_cuda);
    }

    // 7. Clean up and free all allocated resources.
    heterosta_free(sta);

    return 0;
}